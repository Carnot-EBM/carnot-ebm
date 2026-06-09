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

# level 0 -> finish quickly
t0 = None
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        t0 = v["gosubdcyegamj"]
        r = v["roduyfsmiznvg"]
        dy = t0.y - r.y
        dx = t0.x - r.x
        for j, p in enumerate(v["lecfirgqbwunn"]):
            _click(env, p.y + p.height//2, p.x + p.width//2)
            f = _click(env, p.y + p.height//2 + dy, p.x + p.width//2 + dx)
            while getattr(env._game, 'yfbjozweime', False): _click(env, -1, -1)

print("Levels completed after 0:", getattr(f, "levels_completed", 0))

# level 1: SWAP TARGETS!
targets = []
composites = []
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        targets.append(v["gosubdcyegamj"])
        composites.append(v)

# composites[0] goes to targets[1], composites[1] goes to targets[0]
for i in range(len(composites)):
    c = composites[i]
    t = targets[1 - i]
    r = c["roduyfsmiznvg"]
    dy = t.y - r.y
    dx = t.x - r.x
    for p in c["lecfirgqbwunn"]:
        _click(env, p.y + p.height//2, p.x + p.width//2)
        f = _click(env, p.y + p.height//2 + dy, p.x + p.width//2 + dx)
        while getattr(env._game, 'yfbjozweime', False): _click(env, -1, -1)

print("Levels completed after 1:", getattr(f, "levels_completed", 0))
