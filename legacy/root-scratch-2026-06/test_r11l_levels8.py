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
                "centroid": (p.y + p.height // 2, p.x + p.width // 2),
                "obj": p
            }
            pairs.append((p_dict, t_dict))
    return pairs

def _click(env, GameAction, y, x):
    return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})

def run():
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    env = arc.make("r11l-495a7899")
    f = env.reset()
    
    # Large list of offsets
    all_offsets = []
    for dist in range(0, 36, 6):
        if dist == 0:
            all_offsets.append((0, 0))
        else:
            all_offsets.extend([
                (dist, 0), (-dist, 0), (0, dist), (0, -dist),
                (dist, dist), (-dist, -dist), (-dist, dist), (dist, -dist)
            ])
            
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
            ty, tx = t_pairs[0][1]["centroid"]
            
            offset_idx = 0
            for p, t in t_pairs:
                py, px = p["centroid"]
                p_obj = p["obj"]
                
                # Keep trying offsets until piece moves
                moved = False
                while not moved and offset_idx < len(all_offsets):
                    oy, ox = all_offsets[offset_idx]
                    offset_idx += 1
                    
                    _click(env, GameAction, py, px)
                    _click(env, GameAction, ty + oy, tx + ox)
                    
                    while getattr(env._game, 'yfbjozweime', False):
                        env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
                        
                    new_py = p_obj.y + p_obj.height // 2
                    new_px = p_obj.x + p_obj.width // 2
                    
                    if new_py != py or new_px != px:
                        moved = True
                        print(f"    Target {tid} Piece at {py},{px} -> moved to {new_py},{new_px} using offset {oy},{ox}")
                    else:
                        print(f"    Target {tid} Piece at {py},{px} -> failed at offset {oy},{ox}")
                
        lv = int(getattr(env._game, "levels_completed", getattr(f, "levels_completed", 0)))
        state = getattr(f, 'state', None)
        print(f"After level {level} attempt, levels_completed={lv}, state={state}")

run()
