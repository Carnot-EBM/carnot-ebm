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

valid_clicks = []
original_game = copy.deepcopy(env._game)
for action_input in env._game.bmmtkvkbcdd:
    env._game = copy.deepcopy(original_game)
    f2 = env.step(action_input.id, data=action_input.data)
    delta = compute_grid_delta(g, grid_of(f2))
    if delta["n_changed"] > 0:
        valid_clicks.append((action_input.data["x"], action_input.data["y"]))
        
print("Valid clicks:", valid_clicks)
