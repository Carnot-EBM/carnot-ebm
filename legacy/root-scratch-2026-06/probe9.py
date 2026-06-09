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

cpyyshywyc = "pumlzd"
data = game.kacotwgjcyq[cpyyshywyc]
pieces = data["lecfirgqbwunn"]
target = data["gosubdcyegamj"]

tx = target.x + target.width // 2
ty = target.y + target.height // 2

for p in pieces:
    px = p.x + p.width // 2
    py = p.y + p.height // 2
    print(f"Before: piece at {p.x}, {p.y}")
    f = env.step(GameAction.ACTION6, data={"x": px, "y": py})
    f = env.step(GameAction.ACTION6, data={"x": tx, "y": ty})
    
    while game.yfbjozweime:
        f = env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
    print(f"After: piece at {p.x}, {p.y}")

print(f"Target is at {target.x}, {target.y}")
r = data["roduyfsmiznvg"]
print(f"Composite is at {r.x}, {r.y}")
print(f"Composite overlaps target? {r.collides_with(target)}")
print("Level 0 completed?", getattr(f, 'levels_completed', 0))
