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

print("SC25 Test:")
target_key = env._game.jlpticwjyvy[0]
target_grid = env._game.zzpoabuniyn[target_key]
print(f"Target pattern: {target_grid}")

curr_grid = env._game.xhhaqjfncnp
print(f"Current pattern: {curr_grid}")

grid_sprites = env._game.smeinfnvmvn
print("Grid sprites:")
for row in grid_sprites:
    for s in row:
        print(f"  {s.x}, {s.y}")

player = getattr(env._game, 'deymuvatgy', None)
exit_sp = getattr(env._game, 'exydhv', None)
print(f"Player: {player}, Exit: {exit_sp}")

# check if we can click to toggle
s = grid_sprites[0][0]
env.step(GameAction.ACTION6, data={"x": int(s.x), "y": int(s.y)})
print(f"Current pattern after click: {env._game.xhhaqjfncnp}")

