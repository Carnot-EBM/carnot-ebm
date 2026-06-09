import sys
from pathlib import Path
REPO = Path().resolve()
sys.path.insert(0, str(REPO / "python"))
from arc_agi import Arcade
from arc_agi.base import OperationMode
arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
env = arc.make("r11l-495a7899")
f = env.reset()

# skip to level 1
for s in env._game.current_level.get_sprites():
    if s.name.startswith("wakneh"):
        print("Level 0 wall:", s.name, s.x, s.y, s.width, s.height)

f = env.step(6, data={"x": 36, "y": 7})
f = env.step(6, data={"x": 21-6, "y": 39})
while env._game.yfbjozweime: env.step(6, data={"x": -1, "y": -1})
f = env.step(6, data={"x": 59, "y": 27})
f = env.step(6, data={"x": 21+6, "y": 39})
while env._game.yfbjozweime: env.step(6, data={"x": -1, "y": -1})

for s in env._game.current_level.get_sprites():
    if s.name.startswith("wakneh"):
        print("Level 1 wall:", s.name, s.x, s.y, s.width, s.height)

