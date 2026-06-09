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

# select a piece
p = env._game.bbijaigbknc[1]
_click(env, p.y + p.height//2, p.x + p.width//2)

# inspect what happens on click 50, 50
shpscnkkub, hypfkfzmjk = 50, 50
xszukfqfur = env._game.camera.display_to_grid(shpscnkkub, hypfkfzmjk)
mbgbsxgaglu, hanvyecyntc = xszukfqfur
nzdfcwudld = None
for i, njqtixodnb in enumerate(env._game.bbijaigbknc):
    if njqtixodnb.x <= mbgbsxgaglu < njqtixodnb.x + njqtixodnb.width and njqtixodnb.y <= hanvyecyntc < njqtixodnb.y + njqtixodnb.height:
        nzdfcwudld = njqtixodnb
        break
print("nzdfcwudld is", nzdfcwudld)

wiayqaumjug = env._game.wiayqaumjug
sbrblfpykl = wiayqaumjug.width // 2
vdrreavphg = wiayqaumjug.height // 2
wkrkdqxmja = mbgbsxgaglu - sbrblfpykl
adpghqxqvs = hanvyecyntc - vdrreavphg
collides = env._game.gabrtablhx(wkrkdqxmja, adpghqxqvs)
print("gabrtablhx returns", collides)

f = _click(env, 50, 50)
print("yfbjozweime after click:", env._game.yfbjozweime)
