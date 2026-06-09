import sys
from pathlib import Path
REPO = Path().resolve()
sys.path.insert(0, str(REPO / "python"))
from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine.enums import GameAction
arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
env = arc.make("r11l-495a7899")
f = env.reset()

def _click(env, y, x):
    return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})

# level 0
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        ty = v["gosubdcyegamj"].y + v["gosubdcyegamj"].height//2
        tx = v["gosubdcyegamj"].x + v["gosubdcyegamj"].width//2
        for i, p in enumerate(v["lecfirgqbwunn"]):
            py = p.y + p.height//2
            px = p.x + p.width//2
            _click(env, py, px)
            oy = [-6, 6, 0, 0][i%4]
            ox = [0, 0, -6, 6][i%4]
            f = _click(env, ty+oy, tx+ox)

while getattr(env._game, 'yfbjozweime', False):
    f = _click(env, -1, -1)
print("Levels completed:", getattr(f, "levels_completed", 0))

# print pairs in level 1
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        print("Target:", v["gosubdcyegamj"].width, v["gosubdcyegamj"].height)
        for p in v["lecfirgqbwunn"]:
            print("  Piece:", p.width, p.height)
