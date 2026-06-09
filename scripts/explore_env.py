import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
ENVDIR = str(REPO / "environment_files")

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine.enums import GameAction

arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
env = arc.make(sys.argv[1])
f = env.reset()
print(f"Game: {sys.argv[1]}")
print(dir(env._game))
for k, v in env._game.__dict__.items():
    if isinstance(v, (list, dict, set)) and len(v) > 0:
        print(f"{k} (len {len(v)}): {type(v)}")
        try:
            print(f"  sample: {list(v)[0] if isinstance(v, (list, set)) else next(iter(v.items()))}")
        except:
            pass
