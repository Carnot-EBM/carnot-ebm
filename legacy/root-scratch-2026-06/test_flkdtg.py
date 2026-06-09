import sys
from pathlib import Path
REPO = Path().resolve()
sys.path.insert(0, str(REPO / "python"))
from arc_agi import Arcade
from arc_agi.base import OperationMode
arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
env = arc.make("r11l-495a7899")
f = env.reset()

print("Level 0 sprites starting with flkdtg:")
for s in env._game.sprites:
    if s.__class__.__name__.startswith("Flkdtg") or "flkdtg" in s.__class__.__name__.lower():
        print(s.__class__.__name__, s.y, s.x, s.width, s.height)

print("Level 0 sprites starting with roefwu:")
for s in env._game.sprites:
    if s.__class__.__name__.startswith("Roefwu") or "roefwu" in s.__class__.__name__.lower():
        print(s.__class__.__name__, s.y, s.x, s.width, s.height, s.color if hasattr(s, 'color') else 'no color', getattr(s, 'color_remap', 'no remap'))

