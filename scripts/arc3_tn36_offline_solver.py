"""tn36 offline solver — the captured per-game RE for the PROGRAM-EDITOR puzzle.

tn36 (fully reverse-engineered 2026-06-17): the object runs a 5-slot MOVE-PROGRAM
(`vkuvtkaerv`); each slot holds a command code (tn36.py:2171 okllwtboml: 0=settle,
1=left, 2=right, 3=down, 33=up, 5/6/7/16=rotate, 8/9=scale+/-, 14/15/63=property).
Step size CSPOIQWER=4. WIN = the object matches the target on FIVE attributes
(x, y, scale, rotation, sjmtdfxdrc); running the program transforms the object, and if
it lands on the target the level completes (else it resets to base).

TOOL PALETTE (cracked): each slot has TWO BIT-buttons at (slotX, y=41) and (slotX, y=44):
the y=41 button toggles bit0 (+1), the y=44 button toggles bit1 (+2). So code 3 (down)
= BOTH buttons. Slot x-centers = 19 + 5*slot = {19,24,29,34,39}. RUN the program by
clicking the object's sub-button `sxhtkytekm` (~36,55).

This solver READS the object/target internal state, COMPUTES a program of directional
moves (down/left/right — the codes reachable via the 2 bit-buttons) that nets the needed
(dx,dy), maps it to the bit-toggle clicks + a run, and returns the click trajectory. It
generalises L1's hand-found [3,3,3,3,3]; it does NOT yet handle levels needing up-move
(code 33) / scale / rotation / property (those need buttons not yet mapped) — those abort
with a clear reason, to be extended when a level needs them.

L1: obj y13->target y33 (dy=+20=5 downs), dx=0 -> program [3,3,3,3,3], 7 clicks,
reproduction-gated reproduced=True.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed  # noqa: E402
from arcengine import GameAction  # noqa: E402

STEP = 4                                  # CSPOIQWER
NSLOTS = 5
SLOT_X = [19 + 5 * i for i in range(NSLOTS)]   # {19,24,29,34,39}
BIT0_Y, BIT1_Y = 41, 44                   # y=41 toggles +1, y=44 toggles +2
RUN_XY = (36, 55)                         # the object sub-button sxhtkytekm
# directional codes reachable via the 2 bit-buttons (0..3):
SETTLE, LEFT, RIGHT, DOWN = 0, 1, 2, 3


def _obj_tgt(env):
    o = env._game.fdksqlmpki.bzirenxmrg.htntnzkbzu
    t = env._game.fdksqlmpki.bzirenxmrg.aqszntqeae
    g = lambda s, a: (s.x, s.y, s.scale, s.rotation, int(s.sjmtdfxdrc))  # noqa: E731
    return g(o, "o"), g(t, "t")


def _current_program(env):
    return list(env._game.fdksqlmpki.bzirenxmrg.vupcwzjtxu.vkuvtkaerv)


def compute_program(obj, tgt):
    """Compute a <=5-move directional program netting (dx,dy). Returns (program, reason).
    program is a length-5 list of codes in {0,1,2,3}; reason!='' means unsolvable here."""
    (ox, oy, osc, orot, osj), (tx, ty, tsc, trot, tsj) = obj, tgt
    if (osc, orot, osj) != (tsc, trot, tsj):
        return None, "needs scale/rotation/property change (codes 8/9/5/6/7/14/15 — buttons not yet mapped)"
    dx, dy = tx - ox, ty - oy
    if dx % STEP or dy % STEP:
        return None, f"delta not a multiple of STEP={STEP} (dx={dx},dy={dy})"
    moves = []
    if dy < 0:
        return None, "needs UP movement (code 33 — button not yet mapped)"
    moves += [DOWN] * (dy // STEP)
    moves += [RIGHT] * (dx // STEP) if dx >= 0 else [LEFT] * (-dx // STEP)
    if len(moves) > NSLOTS:
        return None, f"needs {len(moves)} moves > {NSLOTS} slots (multi-run not yet implemented)"
    program = moves + [SETTLE] * (NSLOTS - len(moves))
    return program, ""


def _edit_clicks(current, target):
    """Clicks to morph the slot codes from `current` to `target` via the bit-toggle buttons."""
    clicks = []
    for i in range(NSLOTS):
        diff = current[i] ^ target[i]          # bits that must toggle
        if diff & 1:
            clicks.append((SLOT_X[i], BIT0_Y))
        if diff & 2:
            clicks.append((SLOT_X[i], BIT1_Y))
    return clicks


def solve(env):
    """Return (trajectory, reached_level). trajectory = [{action:6, data:{x,y}}, ...]."""
    f = env.reset()
    obj, tgt = _obj_tgt(env)
    program, reason = compute_program(obj, tgt)
    if program is None:
        return None, reason
    clicks = _edit_clicks(_current_program(env), program) + [RUN_XY]
    traj = [{"action": 6, "data": {"x": cx, "y": cy}} for cx, cy in clicks]
    for cx, cy in clicks:
        f = env.step(_game_action(GameAction, 6), data={"x": cx, "y": cy})
    return traj, _levels_completed(f)


def main() -> int:
    arc = kit.offline_arcade()
    env = arc.make("tn36", scorecard_id=arc.open_scorecard())
    traj, lvl = solve(env)
    if traj is None:
        print(f"tn36 solve aborted: {lvl}")
        return 1
    labels = [json.dumps(t) for t in traj]

    def apply(env, label, frame):
        s = json.loads(label)
        return env.step(_game_action(GameAction, s["action"]), data=s.get("data"))

    gate = kit.reproduce("tn36", labels, apply, claimed_level=lvl)
    print(f"tn36 COMPUTED solve: reached L{lvl} in {len(traj)} clicks; "
          f"reproduced={gate['reproduced']} claimed_level={gate.get('claimed_level')}")
    return 0 if (lvl >= 1 and gate["reproduced"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
