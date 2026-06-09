import sys
from pathlib import Path
REPO = Path().resolve()
sys.path.insert(0, str(REPO / "python"))
from arc_agi import Arcade
from arc_agi.base import OperationMode
arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
env = arc.make("r11l-495a7899")
f = env.reset()
from arcengine.enums import GameAction
def _click(env, y, x): return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
from carnot.agentic.arc_agi3_world_model import grid_of
offsets = [(-6, 0), (6, 0), (0, -6), (0, 6), (-6, -6), (6, 6)]
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        t0 = v["gosubdcyegamj"]
        for j, p in enumerate(v["lecfirgqbwunn"]):
            _click(env, p.y + p.height//2, p.x + p.width//2)
            f = _click(env, t0.y + t0.height//2 + offsets[j][1], t0.x + t0.width//2 + offsets[j][0])
while getattr(env._game, 'yfbjozweime', False): f = _click(env, -1, -1)

grid = grid_of(f)
print("Level 1 pieces and targets from grid:")
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        t = v["gosubdcyegamj"]
        print("Target at", t.y, t.x, t.height, t.width)
        # extract grid patch
        patch = grid[t.y:t.y+t.height, t.x:t.x+t.width]
        print(patch)
        for p in v["lecfirgqbwunn"]:
            print(" Piece at", p.y, p.x, p.height, p.width)
            patch = grid[p.y:p.y+p.height, p.x:p.x+p.width]
            print(patch)
