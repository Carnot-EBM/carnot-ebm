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

# skip to level 1
t0 = None
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        t0 = v["gosubdcyegamj"]
        for j, p in enumerate(v["lecfirgqbwunn"]):
            _click(env, p.y + p.height//2, p.x + p.width//2)
            offs = [(-6, 0), (6, 0)]
            _click(env, t0.y + t0.height//2 + offs[j][1], t0.x + t0.width//2 + offs[j][0])
            while getattr(env._game, 'yfbjozweime', False): _click(env, -1, -1)

# try moving ONE piece in level 1
p = env._game.bbijaigbknc[1]  # second piece, not auto-selected
print("Selected piece at start of Level 1:", env._game.wiayqaumjug.y, env._game.wiayqaumjug.x)

print(f"Clicking on piece at {p.y + p.height//2}, {p.x + p.width//2}")
f = _click(env, p.y + p.height//2, p.x + p.width//2)
print("Selected piece after click:", env._game.wiayqaumjug.y, env._game.wiayqaumjug.x)

print("Clicking at target 50, 50")
f = _click(env, 50, 50)
print("yfbjozweime:", env._game.yfbjozweime)

while getattr(env._game, 'yfbjozweime', False): _click(env, -1, -1)
print("Piece pos after animation:", p.y, p.x)

