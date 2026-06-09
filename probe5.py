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
for s in env._game.current_level.get_sprites():
    if 'flkdtg' in s.name or 'roefwulewcui' in s.name or 'roefwu' in s.name:
        print(s.name)
        print(s.pixels)
