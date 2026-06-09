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

def get_valid_offsets(N, ty, tx, margin=6):
    offsets = []
    if N % 2 != 0:
        offsets.append((0, 0))
    dist = 6
    while len(offsets) < N:
        # Try all 4 symmetric pairs at this distance
        pairs_to_try = [
            ((0, dist), (0, -dist)),
            ((dist, 0), (-dist, 0)),
            ((dist, dist), (-dist, -dist)),
            ((-dist, dist), (dist, -dist))
        ]
        
        for p1, p2 in pairs_to_try:
            if len(offsets) >= N: break
            # Check if both are valid
            oy1, ox1 = p1
            oy2, ox2 = p2
            if (margin <= ty + oy1 <= 64 - margin and margin <= tx + ox1 <= 64 - margin and
                margin <= ty + oy2 <= 64 - margin and margin <= tx + ox2 <= 64 - margin):
                offsets.extend([p1, p2])
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
            N = len(t_pairs)
            ty, tx = t_pairs[0][1]["centroid"]
            offsets = get_valid_offsets(N, ty, tx)
            print(f"  Target {tid} (N={N}) at {ty},{tx}")
            
            for i, (p, t) in enumerate(t_pairs):
                oy, ox = offsets[i]
                py, px = p["centroid"]
                print(f"    Piece at {py},{px} -> place at {ty+oy},{tx+ox} (offset {oy},{ox})")
                f = _click(env, GameAction, py, px)
                f = _click(env, GameAction, ty + oy, tx + ox)
                while getattr(env._game, 'yfbjozweime', False):
                    f = env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
                
                # Check where piece actually is now
                p_obj = p["obj"]
                new_py = p_obj.y + p_obj.height // 2
                new_px = p_obj.x + p_obj.width // 2
                print(f"      Actually ended up at {new_py},{new_px}")
                
        lv = int(getattr(f, "levels_completed", 0) or 0)
        state = getattr(f, 'state', None)
        print(f"After level {level} attempt, levels_completed={lv}, state={state}")

run()
