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
def _click(env, y, x): return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})

# level 0
t0 = None
pieces0 = []
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        t0 = v["gosubdcyegamj"]
        pieces0.extend(v["lecfirgqbwunn"])

for j, p in enumerate(pieces0):
    _click(env, p.y + p.height//2, p.x + p.width//2)
    _click(env, t0.y + t0.height//2 + [-6, 6, 0, 0][j%4], t0.x + t0.width//2 + [0, 0, -6, 6][j%4])
while getattr(env._game, 'yfbjozweime', False): _click(env, -1, -1)

# level 1
targets = []
pieces = []
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        targets.append(v["gosubdcyegamj"])
        pieces.extend(v["lecfirgqbwunn"])

# place 3 pieces on target 0, 2 pieces on target 1 (arbitrary assignment)
print(f"Level 1: {len(pieces)} pieces, {len(targets)} targets")
for j, p in enumerate(pieces[:3]):
    _click(env, p.y + p.height//2, p.x + p.width//2)
    _click(env, targets[0].y + targets[0].height//2 + [-6, 6, 0, 0][j%4], targets[0].x + targets[0].width//2 + [0, 0, -6, 6][j%4])
for j, p in enumerate(pieces[3:]):
    _click(env, p.y + p.height//2, p.x + p.width//2)
    _click(env, targets[1].y + targets[1].height//2 + [-6, 6, 0, 0][j%4], targets[1].x + targets[1].width//2 + [0, 0, -6, 6][j%4])
while getattr(env._game, 'yfbjozweime', False): f = _click(env, -1, -1)

print("Levels completed:", getattr(f, "levels_completed", 0))

# place pieces according to true grouping
env.reset()
f = env.reset()
# solve level 0
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        t0 = v["gosubdcyegamj"]
        for j, p in enumerate(v["lecfirgqbwunn"]):
            _click(env, p.y + p.height//2, p.x + p.width//2)
            _click(env, t0.y + t0.height//2 + [-6, 6, 0, 0][j%4], t0.x + t0.width//2 + [0, 0, -6, 6][j%4])
while getattr(env._game, 'yfbjozweime', False): f = _click(env, -1, -1)

# solve level 1 true grouping
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        t0 = v["gosubdcyegamj"]
        for j, p in enumerate(v["lecfirgqbwunn"]):
            _click(env, p.y + p.height//2, p.x + p.width//2)
            _click(env, t0.y + t0.height//2 + [-6, 6, 0, 0][j%4], t0.x + t0.width//2 + [0, 0, -6, 6][j%4])
while getattr(env._game, 'yfbjozweime', False): f = _click(env, -1, -1)

print("Levels completed after true grouping:", getattr(f, "levels_completed", 0))
