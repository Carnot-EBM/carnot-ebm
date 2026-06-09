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

def get_target_offsets(pieces, r, g):
    # compute the exact offsets needed to move r to g
    dy = g.y - r.y
    dx = g.x - r.x
    offsets = []
    for p in pieces:
        # the final position of this piece should be its current position + (dy, dx)
        # so the offsets relative to the target's centroid are:
        ty = g.y + g.height // 2
        tx = g.x + g.width // 2
        py = p.y + p.height // 2 + dy
        px = p.x + p.width // 2 + dx
        offsets.append((py - ty, px - tx))
    return offsets

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
# get assignments for level 0
k, v = list(env._game.kacotwgjcyq.items())[0]
if v["gosubdcyegamj"]:
    r = v["roduyfsmiznvg"]
    g = v["gosubdcyegamj"]
    dy = g.y - r.y
    dx = g.x - r.x
    target_coords = []
    for p in v["lecfirgqbwunn"]:
        target_coords.append((p.y + p.height//2 + dy, p.x + p.width//2 + dx))
    play_level(env, {k: target_coords})

print("Levels completed after 0:", getattr(f, "levels_completed", 0))

k1, v1 = list(env._game.kacotwgjcyq.items())[0]
k2, v2 = list(env._game.kacotwgjcyq.items())[1]

def solve_level_1():
    for perm1 in itertools.permutations(v1["lecfirgqbwunn"]):
        for perm2 in itertools.permutations(v2["lecfirgqbwunn"]):
            env.reset()
            # play level 0
            play_level(env, {k: target_coords})
            
            # try level 1
            assignments = {}
            for k, v, perm in [(k1, v1, perm1), (k2, v2, perm2)]:
                r = v["roduyfsmiznvg"]
                g = v["gosubdcyegamj"]
                dy = g.y - r.y
                dx = g.x - r.x
                coords = []
                # map the original pieces to the permuted final positions
                for p in perm:
                    coords.append((p.y + p.height//2 + dy, p.x + p.width//2 + dx))
                assignments[k] = coords
            
            play_level(env, assignments)
            
            if env._game.levels_completed == 2:
                print("Found working assignment for level 1!")
                return assignments

solve_level_1()
