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

def get_target_coords(k, v):
    r = v["roduyfsmiznvg"]
    g = v["gosubdcyegamj"]
    dy = g.y - r.y
    dx = g.x - r.x
    coords = []
    for p in v["lecfirgqbwunn"]:
        coords.append((p.y + p.height//2 + dy, p.x + p.width//2 + dx))
    return coords

def play_level(env, assignments):
    # assignments is a dict: group_key -> list of target_coords (y, x)
    for k, v in env._game.kacotwgjcyq.items():
        if v["gosubdcyegamj"] and v["roduyfsmiznvg"]:
            target_coords = assignments[k]
            for j, p in enumerate(v["lecfirgqbwunn"]):
                _click(env, p.y + p.height//2, p.x + p.width//2)
                _click(env, target_coords[j][0], target_coords[j][1])
                while getattr(env._game, 'yfbjozweime', False):
                    _click(env, -1, -1)

f = env.reset()

# Precompute level 0 assignment
k0, v0 = list(env._game.kacotwgjcyq.items())[0]
level_0_assign = {k0: get_target_coords(k0, v0)}

play_level(env, level_0_assign)
print("Levels completed after 0:", env._game.levels_completed)

# Get level 1 objects
groups = list(env._game.kacotwgjcyq.items())
k1, v1 = groups[0]
k2, v2 = groups[1]

def solve_level_1():
    for perm1 in itertools.permutations(v1["lecfirgqbwunn"]):
        for perm2 in itertools.permutations(v2["lecfirgqbwunn"]):
            env.reset()
            # play level 0
            k0_new, v0_new = list(env._game.kacotwgjcyq.items())[0]
            play_level(env, {k0_new: get_target_coords(k0_new, v0_new)})
            
            # get new Level 1 groups
            g_new = list(env._game.kacotwgjcyq.items())
            k1_new, v1_new = g_new[0]
            k2_new, v2_new = g_new[1]
            
            # match perm1 and perm2 pieces to coords
            assignments = {}
            for k_new, v_new, perm_original in [(k1_new, v1_new, perm1), (k2_new, v2_new, perm2)]:
                r = v_new["roduyfsmiznvg"]
                g = v_new["gosubdcyegamj"]
                dy = g.y - r.y
                dx = g.x - r.x
                coords = []
                for p_orig in perm_original:
                    # we must map p_orig to the NEW piece in v_new that has the same initial pos
                    # because env.reset() creates new Sprite objects!
                    p_new = next(p for p in v_new["lecfirgqbwunn"] if p.x == p_orig.x and p.y == p_orig.y)
                    coords.append((p_new.y + p_new.height//2 + dy, p_new.x + p_new.width//2 + dx))
                assignments[k_new] = coords
            
            play_level(env, assignments)
            
            if env._game.levels_completed == 2:
                print("Found working assignment for level 1!")
                return assignments
    print("No assignment found!")

solve_level_1()
