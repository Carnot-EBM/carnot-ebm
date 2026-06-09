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
print(f"Level {getattr(f, 'level', 0)}")

# Find sprites
sprites = getattr(env._game, 'lkujttxgs', [])
target_zone = getattr(env._game, 'powykypsm', [])[0]

print(f"Sprites: {len(sprites)}, Target Zone: x={target_zone.x}, y={target_zone.y}, w={target_zone.width}, h={target_zone.height}")
for i, s in enumerate(sprites):
    print(f"Sprite {i}: x={s.x}, y={s.y}, color={getattr(env._game, 'kqywaxhmsb', {}).get(s)}")

# Try selecting first sprite and placing in target zone
s = sprites[0]
env.step(GameAction.ACTION6, data={"x": int(s.x), "y": int(s.y)})
# Click middle of target zone
f = env.step(GameAction.ACTION6, data={"x": int(target_zone.x + target_zone.width/2), "y": int(target_zone.y + target_zone.height/2)})

# Fast forward animations
for _ in range(20):
    f = env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
    if getattr(f, 'levels_completed', 0) > 0:
        break

print(f"Levels completed: {f.levels_completed}")
