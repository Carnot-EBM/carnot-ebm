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

def get_offsets(n):
    if n == 1: return [(0, 0)]
    if n == 2: return [(-6, 0), (6, 0)]
    if n == 3: return [(-6, 0), (6, 0), (0, 0)]

# solve level 0
t0 = None
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        t0 = v["gosubdcyegamj"]
        offs = get_offsets(len(v["lecfirgqbwunn"]))
        for j, p in enumerate(v["lecfirgqbwunn"]):
            _click(env, p.y + p.height//2, p.x + p.width//2)
            f = _click(env, t0.y + t0.height//2 + offs[j][1], t0.x + t0.width//2 + offs[j][0])
            while getattr(env._game, 'yfbjozweime', False):
                f = _click(env, -1, -1)

# level 1
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        t0 = v["gosubdcyegamj"]
        offs = get_offsets(len(v["lecfirgqbwunn"]))
        for j, p in enumerate(v["lecfirgqbwunn"]):
            _click(env, p.y + p.height//2, p.x + p.width//2)
            f = _click(env, t0.y + t0.height//2 + offs[j][1], t0.x + t0.width//2 + offs[j][0])
            while getattr(env._game, 'yfbjozweime', False):
                f = _click(env, -1, -1)

print("Level 1 collision check:")
for cpyyshywyc, data in env._game.kacotwgjcyq.items():
    r = data["roduyfsmiznvg"]
    g = data["gosubdcyegamj"]
    if r and g:
        print(cpyyshywyc, "collides?", r.collides_with(g))
        print("  r pos:", r.y, r.x, "g pos:", g.y, g.x)
