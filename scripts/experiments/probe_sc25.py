import sys
from pathlib import Path
import copy
REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
sys.path.insert(0, str(REPO / "python"))
from arc_agi import Arcade
from arc_agi.base import OperationMode
from carnot.agentic.arc_agi3_world_model import objects, compute_grid_delta, grid_of
from arcengine.enums import GameAction

env = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files")).make("sc25-635fd71a")
f = env.reset()
g = grid_of(f)
objs = objects(g)
print("Objects:", objs)

# Let's see if clicking anywhere does something
valid_clicks = []
original_game = copy.deepcopy(env._game)
for y in range(0, 100, 5):
    for x in range(0, 100, 5):
        env._game = copy.deepcopy(original_game)
        f2 = env.step(GameAction.ACTION6, data={"x": x, "y": y})
        delta = compute_grid_delta(g, grid_of(f2))
        if delta["n_changed"] > 0:
            valid_clicks.append((y, x))
            
print("Valid clicks:", valid_clicks)
