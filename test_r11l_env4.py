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
pieces = []
targets = []
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        ty = v["gosubdcyegamj"].y + v["gosubdcyegamj"].height//2
        tx = v["gosubdcyegamj"].x + v["gosubdcyegamj"].width//2
        targets.append((ty, tx))
        for p in v["lecfirgqbwunn"]:
            py = p.y + p.height//2
            px = p.x + p.width//2
            pieces.append((py, px))

print("Level 0 pieces", pieces, "targets", targets)
for i, p in enumerate(pieces):
    _click(env, p[0], p[1])
    oy = [-6, 6, 0, 0][i%4]
    ox = [0, 0, -6, 6][i%4]
    f = _click(env, targets[0][0]+oy, targets[0][1]+ox)
while getattr(env._game, 'yfbjozweime', False):
    f = _click(env, -1, -1)

# print pairs in level 1
pieces = []
targets = []
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        ty = v["gosubdcyegamj"].y + v["gosubdcyegamj"].height//2
        tx = v["gosubdcyegamj"].x + v["gosubdcyegamj"].width//2
        targets.append((ty, tx))
        for p in v["lecfirgqbwunn"]:
            py = p.y + p.height//2
            px = p.x + p.width//2
            pieces.append((py, px))

print("Level 1 pieces", pieces, "targets", targets)

# place piece 0 on target 1 (wrong)
_click(env, pieces[0][0], pieces[0][1])
f = _click(env, targets[1][0]-6, targets[1][1])
print("After wrong place, levels_completed=", getattr(f, "levels_completed", 0), "state=", getattr(f, "state", None))
