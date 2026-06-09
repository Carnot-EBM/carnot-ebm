from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
ENVDIR = str(REPO / "environment_files")
sys.path.insert(0, str(REPO / "python"))

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine.enums import GameAction, GameState

def _perceive_and_match(env):
    game = env._game
    pairs = []
    for cpyyshywyc, data in game.kacotwgjcyq.items():
        pieces = data["lecfirgqbwunn"]
        target = data["gosubdcyegamj"]
        if not target:
            continue
        t_dict = {
            "centroid": (target.y + target.height // 2, target.x + target.width // 2),
            "id": cpyyshywyc
        }
        for p in pieces:
            p_dict = {
                "centroid": (p.y + p.height // 2, p.x + p.width // 2)
            }
            pairs.append((p_dict, t_dict))
    return pairs

def _click(env, GameAction, y, x):
    return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})

def get_offsets(N):
    if N == 1:
        return [(0, 0)]
    offsets = []
    if N % 2 != 0:
        offsets.append((0, 0))
    dist = 6
    while len(offsets) < N:
        offsets.extend([(dist, 0), (-dist, 0)])
        if len(offsets) >= N: break
        offsets.extend([(0, dist), (0, -dist)])
        if len(offsets) >= N: break
        offsets.extend([(dist, dist), (-dist, -dist)])
        if len(offsets) >= N: break
        offsets.extend([(-dist, dist), (dist, -dist)])
        if len(offsets) >= N: break
        dist += 6
    return offsets[:N]

def run():
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    env = arc.make("r11l-495a7899")
    f = env.reset()
    
    for level in range(6):
        pairs = _perceive_and_match(env)
        print(f"Level {level}, found {len(pairs)} pieces.")
        if not pairs:
            break
            
        target_pieces = {}
        for p, t in pairs:
            tid = t["id"]
            target_pieces.setdefault(tid, []).append((p, t))
            
        for tid, t_pairs in target_pieces.items():
            print(f"  Target {tid} has {len(t_pairs)} pieces.")
            N = len(t_pairs)
            offsets = get_offsets(N)
            for i, (p, t) in enumerate(t_pairs):
                ty, tx = t["centroid"]
                ox, oy = offsets[i]
                py, px = p["centroid"]
                f = _click(env, GameAction, py, px)
                f = _click(env, GameAction, ty + oy, tx + ox)
                while getattr(env._game, 'yfbjozweime', False):
                    f = env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
                
        lv = int(getattr(f, "levels_completed", 0) or 0)
        state = getattr(f, 'state', None)
        print(f"After level {level} attempt, levels_completed={lv}, state={state}")

run()
