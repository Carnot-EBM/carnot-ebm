import sys
sys.path.insert(0, 'python')
from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine.enums import GameAction
from carnot.agentic.arc_agi3_world_model import grid_of
import numpy as np

arc = Arcade(arc_api_key='', operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
env = arc.make('r11l-495a7899')

f = env.reset()
game = env._game

for cpyyshywyc, data in game.kacotwgjcyq.items():
    pieces = data["lecfirgqbwunn"]
    target = data["gosubdcyegamj"]
    
    tx = target.x + target.width // 2
    ty = target.y + target.height // 2
    
    for p in pieces:
        px = p.x + p.width // 2
        py = p.y + p.height // 2
        env.step(GameAction.ACTION6, data={"x": px, "y": py})
        env.step(GameAction.ACTION6, data={"x": tx, "y": ty})
        
        for _ in range(20):
            f = env.step(GameAction.ACTION6, data={"x": -1, "y": -1})

print("Level 0 completed?", getattr(f, 'levels_completed', 0))
