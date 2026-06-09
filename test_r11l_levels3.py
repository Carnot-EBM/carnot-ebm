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
            
        # Just advance by failing intentionally so we can see other levels?
        # No, we can't advance if we fail. 

run()
