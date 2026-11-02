import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
ENVDIR = str(REPO / "environment_files")

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine.enums import GameAction

arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
env = arc.make('sc25-635fd71a')
f = env.reset()

print("Sprites of len 1:")
for k, v in env._game.__dict__.items():
    if isinstance(v, list) and len(v) == 1 and hasattr(v[0], 'tag'):
        print(f"  {k}: tag={v[0].tag}, x={v[0].x}, y={v[0].y}")

