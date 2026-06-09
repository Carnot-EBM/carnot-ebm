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

for level in range(6):
    print("Solving level", level)
    for k, v in env._game.kacotwgjcyq.items():
        if v["gosubdcyegamj"] and v["roduyfsmiznvg"]:
            r = v["roduyfsmiznvg"]
            g = v["gosubdcyegamj"]
            dy = g.y - r.y
            dx = g.x - r.x
            for p in v["lecfirgqbwunn"]:
                f = _click(env, p.y + p.height//2, p.x + p.width//2)
                f = _click(env, p.y + p.height//2 + dy, p.x + p.width//2 + dx)
                while getattr(env._game, 'yfbjozweime', False): f = _click(env, -1, -1)
    
    print("Levels completed:", getattr(f, "levels_completed", 0))
    if getattr(f, "levels_completed", 0) <= level:
        print("Failed to win level", level)
        for k, v in env._game.kacotwgjcyq.items():
            if v["gosubdcyegamj"]:
                r = v["roduyfsmiznvg"]
                g = v["gosubdcyegamj"]
                print(k, "collides?", r.collides_with(g), "r:", r.y, r.x, "g:", g.y, g.x)
        break
