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

def get_click_pos(p_obj):
    for y in range(p_obj.height):
        for x in range(p_obj.width):
            if p_obj.pixels[y, x] != -1 and p_obj.pixels[y, x] != 0:
                return p_obj.y + y, p_obj.x + x
    return p_obj.y + p_obj.height // 2, p_obj.x + p_obj.width // 2

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
                "click_pos": get_click_pos(p),
                "obj": p
            }
            pairs.append((p_dict, t_dict))
    return pairs

def _click(env, GameAction, y, x):
    return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})

def is_valid_click_and_centroid(env, p_obj, click_y, click_x):
    old_x, old_y = p_obj.x, p_obj.y
    p_obj.set_position(click_x, click_y)
    valid = True
    for w in env._game.tdriqoljcbs:
        if p_obj.collides_with(w):
            valid = False
            break
    if valid:
        final_x = click_x - p_obj.width // 2
        final_y = click_y - p_obj.height // 2
        p_obj.set_position(final_x, final_y)
        for w in env._game.tdriqoljcbs:
            if p_obj.collides_with(w):
                valid = False
                break
        if final_x < 0 or final_y < 0 or final_x + p_obj.width > 64 or final_y + p_obj.height > 64:
            valid = False
    p_obj.set_position(old_x, old_y)
    return valid

def generate_zero_sum_combinations(N):
    candidates = [(0,0), (6,0), (-6,0), (0,6), (0,-6), (6,6), (-6,-6), (-6,6), (6,-6), 
                  (12,0), (-12,0), (0,12), (0,-12), (12,12), (-12,-12), (-12,12), (12,-12),
                  (18,0), (-18,0), (0,18), (0,-18)]
    
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
            
            best_combo = None
            for combo in generate_zero_sum_combinations(N):
                valid = True
                placed = []
                for i, (p, t) in enumerate(t_pairs):
                    oy, ox = combo[i]
                    click_y = ty + oy
                    click_x = tx + ox
                    if not is_valid_click_and_centroid(env, p["obj"], click_y, click_x):
                        valid = False
                        break
                    
                    hit_placed = False
                    for (placed_obj, placed_y, placed_x) in placed:
                        top_y = placed_y - placed_obj.height // 2
                        top_x = placed_x - placed_obj.width // 2
                        if top_y <= click_y < top_y + placed_obj.height and top_x <= click_x < top_x + placed_obj.width:
                            if placed_obj.pixels[click_y - top_y, click_x - top_x] != 0 and placed_obj.pixels[click_y - top_y, click_x - top_x] != -1:
                                hit_placed = True
                                break
                    if hit_placed:
                        valid = False
                        break
                        
                    placed.append((p["obj"], click_y, click_x))
                    
                if valid:
                    best_combo = combo
                    break
            
            if not best_combo:
                print(f"  Target {tid} NO VALID COMBO FOUND")
                continue
                
            print(f"  Target {tid} using combo {best_combo}")
            for i, (p, t) in enumerate(t_pairs):
                oy, ox = best_combo[i]
                p_obj = p["obj"]
                click_y, click_x = p["click_pos"]
                
                # To place piece so its centroid is at ty+oy, tx+ox, 
                # we must click the coordinate such that the top-left will be placed
                # NO! Wait! In `gabrtablhx` it sets top-left to `click_x, click_y`!
                # But what if I just use the logic that solved Level 0!
                # In Level 0, we clicked `ty+oy, tx+ox` and it worked perfectly!
                # So we just click `ty+oy, tx+ox`!
                
                _click(env, GameAction, click_y, click_x)
                _click(env, GameAction, ty + oy, tx + ox)
                
                while getattr(env._game, 'yfbjozweime', False):
                    env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
            
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
