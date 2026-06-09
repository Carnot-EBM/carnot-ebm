import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
ENVDIR = str(REPO / "environment_files")

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine.enums import GameAction

arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
env = arc.make('lp85-305b61c3')
f = env.reset()

print("Sprites with 'goal' in name:")
for s in env._game.current_level.sprites:
    if 'goal' in s.name:
        print(f"  {s.name} at {s.y},{s.x}")
    elif s.name in ['bghvgbtwcb', 'fdgmtkfrxl']:
        print(f"  {s.name} at {s.y},{s.x}")

