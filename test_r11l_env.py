import sys
from pathlib import Path
REPO = Path().resolve()
sys.path.insert(0, str(REPO / "python"))
from arc_agi import Arcade
from arc_agi.base import OperationMode
arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
env = arc.make("r11l-495a7899")
f = env.reset()
print(f"Level 0 components:")
for k, v in env._game.kacotwgjcyq.items():
    print(" Target:", getattr(v["gosubdcyegamj"], "width", None), getattr(v["gosubdcyegamj"], "height", None))
    for p in v["lecfirgqbwunn"]:
        print("  Piece:", p.width, p.height)
