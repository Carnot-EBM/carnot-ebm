import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
ENVDIR = str(REPO / "environment_files")

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine.enums import GameAction

arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
env = arc.make('su15-1944f8ab')
f = env.reset()

sprites = getattr(env._game, 'lkujttxgs', [])
target_zone = getattr(env._game, 'powykypsm', [])[0]
s = sprites[0]
print(f"Initial: {s.x}, {s.y}")
env.step(GameAction.ACTION6, data={"x": int(s.x), "y": int(s.y)})
env.step(GameAction.ACTION6, data={"x": int(target_zone.x + target_zone.width/2), "y": int(target_zone.y + target_zone.height/2)})

for i in range(10):
    env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
    print(f"Frame {i}: {s.x}, {s.y}")
