import sys
sys.path.insert(0, 'python')
from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine.enums import GameAction
from carnot.agentic.arc_agi3_world_model import grid_of
import numpy as np

arc = Arcade(arc_api_key='', operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
env = arc.make('r11l-495a7899')

for level_idx in range(6):
    while True:
        f = env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
        if getattr(f, "state", None) is not None:
            break
        # wait wait
        break
    env._game.set_level(level_idx)
    f = env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
    grid = grid_of(f)
    vals, counts = np.unique(grid, return_counts=True)
    print(f"Level {level_idx} Grid colors:", dict(zip(vals, counts)))

