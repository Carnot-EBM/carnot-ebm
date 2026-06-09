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

game = env._game

target_key = game.jlpticwjyvy[0]
target_grid = game.zzpoabuniyn[target_key]
print(f"Target pattern: {target_grid}")

# click to match
for i in range(3):
    for j in range(3):
        if target_grid[i][j]:
            click_x = 25 + j * 5
            click_y = 50 + i * 5
            env.step(GameAction.ACTION6, data={"x": click_x, "y": click_y})

# wait for clicks to process
for _ in range(5):
    env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
print(f"Current pattern after clicks: {game.xhhaqjfncnp}")

# move player
player = game.plnqvukupu
exit_sp_list = game.current_level.get_sprites_by_name("exydhv")
exit_sp = exit_sp_list[0] if exit_sp_list else None

print(f"Player: x={player.x}, y={player.y}")
print(f"Exit: x={exit_sp.x}, y={exit_sp.y}")

while player.x != exit_sp.x or player.y != exit_sp.y:
    if player.x < exit_sp.x:
        f = env.step(GameAction.ACTION4)
    elif player.x > exit_sp.x:
        f = env.step(GameAction.ACTION3)
    elif player.y < exit_sp.y:
        f = env.step(GameAction.ACTION2)
    elif player.y > exit_sp.y:
        f = env.step(GameAction.ACTION1)
    
    player = game.plnqvukupu
    if getattr(f, 'levels_completed', 0) > 0:
        print("LEVEL COMPLETED!")
        break
