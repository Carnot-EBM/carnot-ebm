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

offsets = [(-6, 0), (6, 0), (0, -6), (0, 6), (-6, -6), (6, 6)]

# solve level 0 true grouping
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        t0 = v["gosubdcyegamj"]
        for j, p in enumerate(v["lecfirgqbwunn"]):
            _click(env, p.y + p.height//2, p.x + p.width//2)
            f = _click(env, t0.y + t0.height//2 + offsets[j][1], t0.x + t0.width//2 + offsets[j][0])
while getattr(env._game, 'yfbjozweime', False): f = _click(env, -1, -1)
print("Levels completed after 0:", getattr(f, "levels_completed", 0))

# solve level 1 true grouping
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        t0 = v["gosubdcyegamj"]
        for j, p in enumerate(v["lecfirgqbwunn"]):
            _click(env, p.y + p.height//2, p.x + p.width//2)
            f = _click(env, t0.y + t0.height//2 + offsets[j][1], t0.x + t0.width//2 + offsets[j][0])
while getattr(env._game, 'yfbjozweime', False): f = _click(env, -1, -1)
print("Levels completed after 1:", getattr(f, "levels_completed", 0))

# solve level 2 true grouping
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        t0 = v["gosubdcyegamj"]
        for j, p in enumerate(v["lecfirgqbwunn"]):
            _click(env, p.y + p.height//2, p.x + p.width//2)
            f = _click(env, t0.y + t0.height//2 + offsets[j][1], t0.x + t0.width//2 + offsets[j][0])
while getattr(env._game, 'yfbjozweime', False): f = _click(env, -1, -1)
print("Levels completed after 2:", getattr(f, "levels_completed", 0))
