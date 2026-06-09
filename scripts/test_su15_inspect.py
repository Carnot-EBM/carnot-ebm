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

print(f"Goal: {getattr(env._game, 'dsqlbvwaj', [])}")

sprites = getattr(env._game, 'lkujttxgs', [])
target_zone = getattr(env._game, 'powykypsm', [])[0]
print(f"Sprites:")
for s in sprites:
    color = getattr(env._game, 'kqywaxhmsb', {}).get(s)
    print(f"  color={color}, x={s.x}, y={s.y}, w={s.width}, h={s.height}")

s = sprites[0]
print(f"Clicking sprite at x={s.x + s.width//2}, y={s.y + s.height//2}")
env.step(GameAction.ACTION6, data={"x": int(s.x + s.width//2), "y": int(s.y + s.height//2)})
print(f"Clicking target at x={target_zone.x + target_zone.width//2}, y={target_zone.y + target_zone.height//2}")
env.step(GameAction.ACTION6, data={"x": int(target_zone.x + target_zone.width//2), "y": int(target_zone.y + target_zone.height//2)})

for i in range(20):
    f = env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
    print(f"Frame {i}: sprite is at x={s.x}, y={s.y}")
    if getattr(f, 'levels_completed', 0) > 0:
        print("LEVEL COMPLETED!")
        break

