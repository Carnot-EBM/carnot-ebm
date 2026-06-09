import sys
sys.path.insert(0, 'python')
from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine.enums import GameAction
import numpy as np

arc = Arcade(arc_api_key='', operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
env = arc.make('r11l-495a7899')
f = env.reset()
game = env._game

p1, p2 = game.kacotwgjcyq["pumlzd"]["lecfirgqbwunn"]
t = game.kacotwgjcyq["pumlzd"]["gosubdcyegamj"]
r = game.kacotwgjcyq["pumlzd"]["roduyfsmiznvg"]

tx = t.x + t.width // 2
ty = t.y + t.height // 2

p1.set_position(tx - p1.width//2, ty - p1.height//2)
p2.set_position(tx - p2.width//2, ty - p2.height//2)

game.rvkbignsyr(r, [p1, p2])
print("R overlaps T?", r.collides_with(t))

