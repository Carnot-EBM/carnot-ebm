import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO / "python"))

from arc_agi import Arcade
from arc_agi.base import OperationMode
from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash, objects, compute_grid_delta, GameGraph

arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
game_id = "su15-1944f8ab"
env = arc.make(game_id)
f = env.reset()

gg = GameGraph(game_id)
start_grid = grid_of(f)
start_fh = frame_hash(start_grid)
gg.see_node(start_fh, f)

grid_cache = {start_fh: start_grid}

def cands_fn(fh, n):
    grid = grid_cache[fh]
    if grid.size == 0:
        return []
    objs = objects(grid)
    cands = []
    av = n.get("available_actions", [])
    if 6 in av:
        for y, x in objs:
            cands.append((6, y, x))
    for a in av:
        if a != 6:
            cands.append((a,))
    return cands

actions_used = 0
budget = 2000

while actions_used < budget:
    grid = grid_of(f)
    fh = frame_hash(grid)
    if fh not in grid_cache:
        grid_cache[fh] = grid
        
    cands = cands_fn(fh, gg.nodes.get(fh, {"available_actions": getattr(f, "available_actions", [])}))
    untested = gg.untested(fh, cands)
    
    if untested:
        akey = untested[0]
        if len(akey) == 3:
            a_int, y, x = akey
            data = {"y": int(y), "x": int(x)}
        else:
            a_int = akey[0]
            data = {}
            
        f2 = env.step(a_int, data=data)
        actions_used += 1
        
        grid2 = grid_of(f2)
        fh2 = frame_hash(grid2)
        if fh2 not in grid_cache:
            grid_cache[fh2] = grid2
            
        delta = compute_grid_delta(grid, grid2)
        lv = int(getattr(f2, "levels_completed", 0) or 0)
        st = getattr(f2, "state", None)
        go = st in (2, 3) # WIN or GAME_OVER
        
        gg.see_node(fh2, f2)
        ld = lv - int(getattr(f, "levels_completed", 0) or 0)
        gg.record(fh, akey, fh2, delta, ld, go)
        
        if lv > 0:
            print(f"Solved L0 in {actions_used} actions!")
            break
            
        f = f2
        if go:
            f = env.reset()
    else:
        # Navigate to frontier
        frontiers = gg.frontier_states(cands_fn)
        if not frontiers:
            print("No more frontiers!")
            break
        a = gg.shortest_path_action(fh, frontiers)
        if a is None:
            # Maybe game over or blocked, reset
            f = env.reset()
            continue
            
        if len(a) == 3:
            a_int, y, x = a
            data = {"y": int(y), "x": int(x)}
        else:
            a_int = a[0]
            data = {}
            
        f = env.step(a_int, data=data)
        actions_used += 1

print(f"Done. Used {actions_used} actions. Levels completed: {getattr(f, 'levels_completed', 0)}")
