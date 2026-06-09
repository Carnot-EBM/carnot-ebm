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
        for j, p in enumerate(v["lecfirgqbwunn"]):
            _click(env, p.y + p.height//2, p.x + p.width//2)
            offs = [(-6, 0), (6, 0)]
            _click(env, t0.y + t0.height//2 + offs[j][1], t0.x + t0.width//2 + offs[j][0])
            while getattr(env._game, 'yfbjozweime', False): _click(env, -1, -1)

# level 1
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        t0 = v["gosubdcyegamj"]
        n = len(v["lecfirgqbwunn"])
        offs = [(-6, 0), (6, 0), (0, -6), (0, 6), (-6, -6), (6, 6)]
        for j, p in enumerate(v["lecfirgqbwunn"]):
            _click(env, p.y + p.height//2, p.x + p.width//2)
            _click(env, t0.y + t0.height//2 + offs[j][1], t0.x + t0.width//2 + offs[j][0])
            while getattr(env._game, 'yfbjozweime', False): _click(env, -1, -1)

print("Level 1 Final Check:")
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        r = v["roduyfsmiznvg"]
        g = v["gosubdcyegamj"]
        print(f"Target {k}: r pos ({r.y}, {r.x}), g pos ({g.y}, {g.x}), collides: {r.collides_with(g)}")
