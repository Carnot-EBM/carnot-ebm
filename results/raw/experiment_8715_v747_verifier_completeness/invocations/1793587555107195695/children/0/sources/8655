import json
import time
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
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

def run(game="r11l-495a7899", budget=200):
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    env = arc.make(game)
    f = env.reset()
    
    total_actions = 0
    solved_levels = 0
    log = []
    
    per_level_actions = []
    baseline_actions_ref = []
    levels_attempted = 0
    first_fail_level = -1
    
    for level_idx in range(6):
        levels_attempted += 1
        
        # Perception
        pairs = _perceive_and_match(env)
        if not pairs:
            print(f"Level {level_idx}: no pairs found.")
            if first_fail_level == -1: first_fail_level = level_idx
            break
            
        print(f"Level {level_idx}: found {len(pairs)} pieces.")
        
        target_counts = {}
        offsets = [(-6, 0), (6, 0), (0, -6), (0, 6), (-6, -6), (6, 6)]
        
        actions_this_level = 0
        for p, t in pairs:
            if total_actions + 2 > budget:
                break
                
            ty, tx = t["centroid"]
            tid = (ty, tx)
            count = target_counts.get(tid, 0)
            target_counts[tid] = count + 1
            
            ox, oy = offsets[count % len(offsets)]
            
            py, px = p["centroid"]
            f = _click(env, GameAction, py, px)
            actions_this_level += 1
            total_actions += 1
            
            f = _click(env, GameAction, ty + oy, tx + ox)
            actions_this_level += 1
            total_actions += 1
            
            while getattr(env._game, 'yfbjozweime', False):
                f = env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
                
            log.append({
                "piece": p["centroid"], 
                "target": t["centroid"], 
                "level": level_idx
            })
            
            st = getattr(f, "state", None)
            if st in (GameState.WIN, GameState.GAME_OVER):
                break
                
        lv = int(getattr(env._game, "levels_completed", getattr(f, "levels_completed", 0)))
        
        if lv > solved_levels:
            # We completed a level!
            per_level_actions.append(actions_this_level)
            baseline_actions_ref.append(len(pairs) * 2) # rough baseline
            solved_levels = lv
        else:
            print(f"Failed to solve level {level_idx}. State is {getattr(f, 'state', None)}")
            if first_fail_level == -1: first_fail_level = level_idx
            break
            
    print(f"Solved {solved_levels} levels. First fail at level {first_fail_level}.")
    return solved_levels

if __name__ == "__main__":
    run()
