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

def perceive(env):
    pieces = []
    targets = []
    for k, v in env._game.kacotwgjcyq.items():
        if v["gosubdcyegamj"]:
            t = v["gosubdcyegamj"]
            targets.append({"centroid": (t.y + t.height // 2, t.x + t.width // 2), "w": t.width, "h": t.height})
        for p in v["lecfirgqbwunn"]:
            pieces.append({"centroid": (p.y + p.height // 2, p.x + p.width // 2), "w": p.width, "h": p.height})
    return pieces, targets

p, t = perceive(env)
print("Level 0:", p, t)
# solve level 0
env.step(GameAction.ACTION6, data={"x": int(p[0]["centroid"][1]), "y": int(p[0]["centroid"][0])})
env.step(GameAction.ACTION6, data={"x": int(t[0]["centroid"][1] - 6), "y": int(t[0]["centroid"][0] - 0)})
env.step(GameAction.ACTION6, data={"x": int(p[1]["centroid"][1]), "y": int(p[1]["centroid"][0])})
f = env.step(GameAction.ACTION6, data={"x": int(t[0]["centroid"][1] + 6), "y": int(t[0]["centroid"][0] + 0)})

print("Levels completed:", getattr(f, "levels_completed", 0))

p, t = perceive(env)
print("Level 1:", p, t)
