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

print("Buttons:")
buttons = getattr(env._game, 'afhycvvjg', [])
for i, btn in enumerate(buttons):
    print(f"Button {i}: x={btn.x}, y={btn.y}")

def extract_positions(p):
    positions = []
    if isinstance(p, dict):
        for k, v in p.items():
            positions.extend(extract_positions(v))
    elif hasattr(p, 'y') and hasattr(p, 'x'):
        positions.append((p.y, p.x))
    return positions

def state_hash():
    p = getattr(env._game, 'uopmnplcnv', {})
    coords = extract_positions(p)
    return tuple(sorted(coords))

print(f"Initial state hash: {state_hash()}")

# click button 0
env.step(GameAction.ACTION6, data={"x": int(buttons[0].x), "y": int(buttons[0].y)})
print(f"After button 0 click hash: {state_hash()}")

# click button 1
env.step(GameAction.ACTION6, data={"x": int(buttons[1].x), "y": int(buttons[1].y)})
print(f"After button 1 click hash: {state_hash()}")
