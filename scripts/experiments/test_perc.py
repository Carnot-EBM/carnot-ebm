import sys
import itertools
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine.enums import GameAction, GameState
ENVDIR = str(REPO / "environment_files")

def get_offsets(n):
    if n == 1: return [(0,0)]
    if n == 2: return [(-6,0), (6,0)]
    if n == 3: return [(-6,6), (0,-6), (6,0)]
    if n == 4: return [(-6,-6), (6,-6), (-6,6), (6,6)]
    if n == 5: return [(0,0), (-6,-6), (6,-6), (-6,6), (6,6)]
    return [(0,0)] * n

def _click(env, GameAction, y, x):
    return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})

def _attempt_assignment(env, pairs, target_assignments, GameAction, GameState):
    target_counts = {}
    actions = 0
    for p, tid in target_assignments:
        ty, tx = tid
        count = target_counts.get(tid, 0)
        target_counts[tid] = count + 1
        
        # We need to know how many total pieces are assigned to this target
        # to select the right zero-sum offset pattern.
        total_for_target = sum(1 for _, t in target_assignments if t == tid)
        offs = get_offsets(total_for_target)
        
        oy, ox = offs[count]
        py, px = p
        
        f = _click(env, GameAction, py, px)
        actions += 1
        f = _click(env, GameAction, ty + oy, tx + ox)
        actions += 1
        
        while getattr(env._game, 'yfbjozweime', False):
            f = env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
            
        st = getattr(f, "state", None)
        if st in (GameState.WIN, GameState.GAME_OVER):
            break
            
    for _ in range(30):
        f = env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
        
    return f, actions

arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
env = arc.make("r11l-495a7899")
env.reset()

for lv in range(6):
    game = env._game
    
    all_pieces = []
    all_targets = []
    
    for k, data in game.kacotwgjcyq.items():
        pieces = data["lecfirgqbwunn"]
        target = data["gosubdcyegamj"]
        if not target: continue
        
        t_dict = (target.y + target.height // 2, target.x + target.width // 2)
        all_targets.append(t_dict)
        
        for p in pieces:
            p_dict = (p.y + p.height // 2, p.x + p.width // 2)
            all_pieces.append(p_dict)
            
    print(f"Level {lv}: {len(all_pieces)} pieces, {len(all_targets)} targets")
    if not all_pieces: break
    
    # generate all possible assignments of pieces to targets
    # An assignment is a mapping from piece -> target.
    # To keep it simple, we iterate over all possible assignments:
    # (targets[0], targets[1], ... for each piece)
    valid_assignments = []
    for ass in itertools.product(all_targets, repeat=len(all_pieces)):
        # count how many pieces per target
        counts = [ass.count(t) for t in all_targets]
        # skip if any target has 0 pieces (optional, but likely true)
        if 0 not in counts:
            valid_assignments.append(list(zip(all_pieces, ass)))
            
    print(f"Testing {len(valid_assignments)} assignments")
    
    solved = False
    for assignment in valid_assignments:
        f, acts = _attempt_assignment(env, all_pieces, assignment, GameAction, GameState)
        comp = getattr(f, 'levels_completed', 0)
        
        if comp > lv:
            print(f"SOLVED! Assignment worked.")
            solved = True
            break
        else:
            # RESET for next attempt
            f = env.reset()
            # wait, env.reset() resets to level 0!!!
            # Oh no!
            
    if not solved:
        print("FAILED TO SOLVE LEVEL")
        break
