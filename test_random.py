import sys
from pathlib import Path
REPO = Path().resolve()
sys.path.insert(0, str(REPO / "python"))
from arc_agi import Arcade
from arc_agi.base import OperationMode
arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))

env = arc.make("r11l-495a7899")
f = env.reset()
print("Run 1:")
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        print(k, [(p.y, p.x) for p in v["lecfirgqbwunn"]])

f = env.reset()
print("Run 2:")
for k, v in env._game.kacotwgjcyq.items():
    if v["gosubdcyegamj"]:
        print(k, [(p.y, p.x) for p in v["lecfirgqbwunn"]])
