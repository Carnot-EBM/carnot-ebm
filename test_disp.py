import sys
from pathlib import Path
REPO = Path().resolve()
sys.path.insert(0, str(REPO / "python"))
from arc_agi import Arcade
from arc_agi.base import OperationMode
arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
env = arc.make("r11l-495a7899")
f = env.reset()

print("Grid size:", env._game.current_level.grid_size)
print("display_to_grid(50, 50):", env._game.camera.display_to_grid(50, 50))
print("display_to_grid(300, 300):", env._game.camera.display_to_grid(300, 300))
