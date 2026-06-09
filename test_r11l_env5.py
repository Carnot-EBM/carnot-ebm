import sys
from pathlib import Path
REPO = Path().resolve()
sys.path.insert(0, str(REPO / "python"))
from arc_agi import Arcade
from arc_agi.base import OperationMode
arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
env = arc.make("r11l-495a7899")
f = env.reset()

for i in range(2):
    print(f"Level {i}")
    for k, v in env._game.kacotwgjcyq.items():
        if v["gosubdcyegamj"]:
            t = v["gosubdcyegamj"]
            print(" Target:", t.width, t.height, t.color, t.color_remap)
            for p in v["lecfirgqbwunn"]:
                print("  Piece:", p.width, p.height, p.color, p.color_remap)
    if i == 0:
        # solve level 0
        from arcengine.enums import GameAction
        def _click(env, y, x): return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
        for k, v in env._game.kacotwgjcyq.items():
            if v["gosubdcyegamj"]:
                ty = v["gosubdcyegamj"].y + v["gosubdcyegamj"].height//2
                tx = v["gosubdcyegamj"].x + v["gosubdcyegamj"].width//2
                for j, p in enumerate(v["lecfirgqbwunn"]):
                    _click(env, p.y + p.height//2, p.x + p.width//2)
                    _click(env, ty + [-6, 6, 0, 0][j%4], tx + [0, 0, -6, 6][j%4])
        while getattr(env._game, 'yfbjozweime', False):
            _click(env, -1, -1)
