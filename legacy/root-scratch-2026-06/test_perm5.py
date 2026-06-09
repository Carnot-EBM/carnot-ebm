import sys
from pathlib import Path
REPO = Path().resolve()
sys.path.insert(0, str(REPO / "python"))
from arc_agi import Arcade
from arc_agi.base import OperationMode
arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
env = arc.make("r11l-495a7899")
from arcengine.enums import GameAction
import itertools

def _click(env, y, x): return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})

def play_assignments(env, assignments):
    # assignments: dict of k -> list of ((start_y, start_x), (target_y, target_x))
    f = None
    for k, v in env._game.kacotwgjcyq.items():
        if v["gosubdcyegamj"] and v["roduyfsmiznvg"]:
            mapping = assignments[k]
            # match current pieces to mapping by start_pos
            for (sy, sx), (ty, tx) in mapping:
                # find the piece
                p = next(p for p in v["lecfirgqbwunn"] if p.y + p.height//2 == sy and p.x + p.width//2 == sx)
                f = _click(env, sy, sx)
                f = _click(env, ty, tx)
                while getattr(env._game, 'yfbjozweime', False):
                    f = _click(env, -1, -1)
    return f

def solve_all_levels():
    all_assignments = []  # list of level_assignments
    
    for level in range(6):
        print(f"Solving level {level}...")
        # get initial state of this level
        f = env.reset()
        for assign in all_assignments:
            f = play_assignments(env, assign)
        
        start_level_val = getattr(f, "levels_completed", 0)
        
        valid_groups = [(k, v) for k, v in env._game.kacotwgjcyq.items() if v["gosubdcyegamj"] and v["roduyfsmiznvg"]]
        group_perms = []
        for k, v in valid_groups:
            r = v["roduyfsmiznvg"]
            g = v["gosubdcyegamj"]
            dy = g.y - r.y
            dx = g.x - r.x
            
            # get all pieces start centers
            starts = [(p.y + p.height//2, p.x + p.width//2) for p in v["lecfirgqbwunn"]]
            # final offsets
            targets = [(sy + dy, sx + dx) for sy, sx in starts]
            
            # perms of targets
            perms = list(itertools.permutations(targets))
            group_perms.append((k, starts, perms))
            
        all_perms_combinations = list(itertools.product(*[gp[2] for gp in group_perms]))
        print(f" Trying {len(all_perms_combinations)} combinations...")
        
        found = False
        for perm_combo in all_perms_combinations:
            f = env.reset()
            for assign in all_assignments:
                f = play_assignments(env, assign)
            
            level_assignments = {}
            for i, (k, starts, _) in enumerate(group_perms):
                targets = perm_combo[i]
                level_assignments[k] = list(zip(starts, targets))
            
            f = play_assignments(env, level_assignments)
            if getattr(f, "levels_completed", 0) > start_level_val:
                print(f"Level {level} solved!")
                all_assignments.append(level_assignments)
                found = True
                break
                
        if not found:
            print(f"Failed to solve level {level}!")
            return

solve_all_levels()
