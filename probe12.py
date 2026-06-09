import sys
sys.path.insert(0, 'python')
from arc_agi import Arcade
from arc_agi.base import OperationMode

arc = Arcade(arc_api_key='', operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
env = arc.make('r11l-495a7899')

f = env.reset()
env._game.set_level(1)

for s in env._game.current_level.get_sprites():
    if 'roefwulewcui' in s.name:
        print(s.name)
        print(s.pixels)
