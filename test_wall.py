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
print("Level 1 placing pieces:")
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        t0 = v["gosubdcyegamj"]
        n = len(v["lecfirgqbwunn"])
        offs = [(-6, 0), (6, 0), (0, -6), (0, 6), (-6, -6), (6, 6)]
        for j, p in enumerate(v["lecfirgqbwunn"]):
            _click(env, p.y + p.height//2, p.x + p.width//2)
            
            # place piece
            ty = t0.y + t0.height//2 + offs[j][1]
            tx = t0.x + t0.width//2 + offs[j][0]
            
            # compute wkrkdqxmja
            wkrkdqxmja = tx - env._game.wiayqaumjug.width // 2
            adpghqxqvs = ty - env._game.wiayqaumjug.height // 2
            
            collides_wall = env._game.gabrtablhx(wkrkdqxmja, adpghqxqvs)
            print(f"Target {k} piece {j}: click at {ty}, {tx}. Collides wall? {collides_wall}")
            
            _click(env, ty, tx)
            while getattr(env._game, 'yfbjozweime', False): _click(env, -1, -1)

