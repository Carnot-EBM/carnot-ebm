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
print(f"Level {getattr(f, 'level', 0)}")
print(f"Action 6 coords to try...")
buttons = getattr(env._game, 'afhycvvjg', [])
for i, btn in enumerate(buttons):
    print(f"Button {i}: x={btn.x}, y={btn.y}")

f = env.step(GameAction.ACTION6, data={"x": int(buttons[0].x), "y": int(buttons[0].y)})
print(f.levels_completed)
