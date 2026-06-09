import sys
from pathlib import Path
REPO = Path().resolve()
sys.path.insert(0, str(REPO / "python"))
from arc_agi import Arcade
from arc_agi.base import OperationMode
arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
env = arc.make("r11l-495a7899")
f = env.reset()

print("Level 0 properties containing flkdtg:")
for k, v in env._game.__dict__.items():
    if isinstance(v, list) and len(v) > 0:
        if "flkdtg" in v[0].__class__.__name__.lower():
            print(k, "has", len(v), "flkdtg sprites")
            for s in v:
                print(" ", s.y, s.x, s.width, s.height)
    if "roefwu" in k.lower():
        print("found roefwu:", k)

print("Level 0 kacotwgjcyq:")
for k, v in env._game.kacotwgjcyq.items():
    for tk, tv in v.items():
        if isinstance(tv, list):
            for item in tv:
                if "flkdtg" in item.__class__.__name__.lower():
                    print("Found flkdtg in kacotwgjcyq:", tk, item.y, item.x)
        elif tv and "flkdtg" in tv.__class__.__name__.lower():
            print("Found flkdtg in kacotwgjcyq:", tk, tv.y, tv.x)
