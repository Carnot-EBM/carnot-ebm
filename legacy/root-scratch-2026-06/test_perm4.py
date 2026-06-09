import sys
from pathlib import Path
REPO = Path().resolve()
sys.path.insert(0, str(REPO / "python"))
from arc_agi import Arcade
from arc_agi.base import OperationMode
arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
env = arc.make("r11l-495a7899")
f = env.reset()
from arcengine.enums import GameAction
import itertools

def _click(env, y, x): return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})

def get_target_coords(v):
    r = v["roduyfsmiznvg"]
    g = v["gosubdcyegamj"]
    dy = g.y - r.y
    dx = g.x - r.x
    coords = []
    for p in v["lecfirgqbwunn"]:
        coords.append((p.y + p.height//2 + dy, p.x + p.width//2 + dx))
    return coords

def play_level(env, assignments):
    f = None
    for k, v in env._game.kacotwgjcyq.items():
        if v["gosubdcyegamj"] and v["roduyfsmiznvg"]:
            target_coords = assignments[k]
            for j, p in enumerate(v["lecfirgqbwunn"]):
                f = _click(env, p.y + p.height//2, p.x + p.width//2)
                f = _click(env, target_coords[j][0], target_coords[j][1])
                while getattr(env._game, 'yfbjozweime', False):
                    f = _click(env, -1, -1)
    return f

def solve_all_levels():
    env.reset()
    all_assignments = []
    
    for level in range(6):
        # find groups with targets
        valid_groups = [(k, v) for k, v in env._game.kacotwgjcyq.items() if v["gosubdcyegamj"] and v["roduyfsmiznvg"]]
        
        # generate permutations for each group
        group_perms = []
        for k, v in valid_groups:
            perms = list(itertools.permutations(v["lecfirgqbwunn"]))
            group_perms.append((k, v, perms))
        
        # try all combinations of permutations
        # list of (perm_for_group1, perm_for_group2, ...)
        all_perms_combinations = list(itertools.product(*[gp[2] for gp in group_perms]))
        
        found = False
        for perm_combo in all_perms_combinations:
            env.reset()
            # replay previous levels
            for prev_assign in all_assignments:
                # need to map prev_assign to new env instances
                mapped_assign = {}
                for pk, pv in prev_assign.items():
                    # just extract the coords
                    pass
            # Wait, this is getting complicated to replay everything properly.
            pass

solve_all_levels()
