from __future__ import annotations
import sys
import itertools
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
                "centroid": (p.y + p.height // 2, p.x + p.width // 2),
                "obj": p
            }
            pairs.append((p_dict, t_dict))
    return pairs

def _click(env, GameAction, y, x):
    return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})

def is_valid_centroid(env, p_obj, cy, cx):
    old_x, old_y = p_obj.x, p_obj.y
    new_x = cx - p_obj.width // 2
    new_y = cy - p_obj.height // 2
    p_obj.set_position(new_x, new_y)
    valid = True
    for w in env._game.tdriqoljcbs:
        if p_obj.collides_with(w):
            valid = False
            break
    # also check board boundaries
    if new_x < 0 or new_y < 0 or new_x + p_obj.width > 64 or new_y + p_obj.height > 64:
        valid = False
    p_obj.set_position(old_x, old_y)
    return valid

def generate_zero_sum_combinations(N):
    # Candidate offsets
    candidates = [(0,0), (6,0), (-6,0), (0,6), (0,-6), (6,6), (-6,-6), (-6,6), (6,-6), 
                  (12,0), (-12,0), (0,12), (0,-12), (12,12), (-12,-12), (-12,12), (12,-12)]
    
    # We want N unique offsets that sum to (0,0)
    for combo in itertools.combinations(candidates, N):
        sy = sum(oy for oy, ox in combo)
        sx = sum(ox for oy, ox in combo)
        if sy == 0 and sx == 0:
            yield combo

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
            N = len(t_pairs)
            ty, tx = t_pairs[0][1]["centroid"]
            
            # Find a valid zero-sum combination
            best_combo = None
            for combo in generate_zero_sum_combinations(N):
                valid = True
                for i, (p, t) in enumerate(t_pairs):
                    oy, ox = combo[i]
                    if not is_valid_centroid(env, p["obj"], ty + oy, tx + ox):
                        valid = False
                        break
                if valid:
                    best_combo = combo
                    break
            
            if not best_combo:
                continue
                
            for i, (p, t) in enumerate(t_pairs):
                oy, ox = best_combo[i]
                py, px = p["centroid"]
                
                # Check if it's already there
                if py == ty + oy and px == tx + ox:
                    continue
                    
                f = _click(env, GameAction, py, px)
                f = _click(env, GameAction, ty + oy, tx + ox)
                while getattr(env._game, 'yfbjozweime', False):
                    f = env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
            
            # compute average manually
            sum_y, sum_x = 0, 0
            for p, _ in t_pairs:
                p_obj = p["obj"]
                sum_y += p_obj.y + p_obj.height // 2
                sum_x += p_obj.x + p_obj.width // 2
            print(f"  Target {tid} expected {ty},{tx}, actual average {sum_y // N},{sum_x // N}")
                    
        lv = int(getattr(env._game, "levels_completed", getattr(f, "levels_completed", 0)))
        state = getattr(f, 'state', None)
        print(f"After level {level} attempt, levels_completed={lv}, state={state}")

run()
