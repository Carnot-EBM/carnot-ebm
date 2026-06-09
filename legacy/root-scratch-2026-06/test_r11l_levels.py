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
            "centroid": (target.y + target.height // 2, target.x + target.width // 2)
        }
        
        for p in pieces:
            p_dict = {
                "centroid": (p.y + p.height // 2, p.x + p.width // 2)
            }
            pairs.append((p_dict, t_dict))
            
    return pairs

def _click(env, GameAction, y, x):
    return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})

def run():
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    env = arc.make("r11l-495a7899")
    f = env.reset()
    
    for level in range(6):
        pairs = _perceive_and_match(env)
        print(f"Level {level}, found {len(pairs)} pieces.")
        if not pairs:
            print("No pairs found!")
            break
            
        target_counts = {}
        offsets = [(-6, 0), (6, 0), (0, -6), (0, 6), (-6, -6), (6, 6), (-6, 6), (6, -6), (-12, 0), (12, 0), (0, -12), (0, 12)]
        
        for p, t in pairs:
            ty, tx = t["centroid"]
            tid = (ty, tx)
            count = target_counts.get(tid, 0)
            target_counts[tid] = count + 1
            
            ox, oy = offsets[count % len(offsets)]
            
            py, px = p["centroid"]
            f = _click(env, GameAction, py, px)
            f = _click(env, GameAction, ty + oy, tx + ox)
            
            while getattr(env._game, 'yfbjozweime', False):
                f = env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
                
        lv = int(getattr(f, "levels_completed", 0) or 0)
        print(f"After level {level} attempt, levels_completed={lv}, state={getattr(f, 'state', None)}")

run()
