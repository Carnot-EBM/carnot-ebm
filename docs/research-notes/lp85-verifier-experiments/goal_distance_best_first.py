"""lp85 VERIFIER-ROUTED solver: discover -> GOAL-DIRECTED best-first search -> advance.

Replaces blind BFS with greedy best-first guided by a goal-distance verifier:
  heuristic h(grid) = sum over goal-marker cells of (min L1-distance to a movable piece cell).
Goals = invariant cells of the goal-sprite palette {11,12} (goal/goal-o); pieces = cells any button moves.
Best-first explores toward lower h (pieces approaching goals), reaching deep solutions (L2 human=38)
that branching-5 BFS cannot. This is the Carnot verifier-routes-the-search thesis on a live game.
"""
import hashlib, time, json, heapq
import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _game_action,_levels_completed,_game_over
from carnot.agentic.arc_executable_world_model import to_logical,detect_cell

GAME="lp85"; STEP=2; TARGET=8; NODES=12000; DEPTH=80; TIME_BUDGET=900
GOAL_COLORS={11,12}
arc=kit.offline_arcade(); env=arc.make(GAME,scorecard_id=arc.open_scorecard())
f0=env.reset(); cell=detect_cell(grid_of(f0)); H,W=to_logical(grid_of(f0),cell).shape
T0=time.time()

def replay(seq):
    f=env.reset()
    for a in seq:
        f=env.step(_game_action(GameAction,6),data=a)
        if f is None: return None,-1,True
        if _game_over(f): return to_logical(grid_of(f),cell),_levels_completed(f),True
    return to_logical(grid_of(f),cell),_levels_completed(f),False

def ghash(g): return hashlib.md5(g.tobytes()).hexdigest()[:12]

def discover(prefix):
    base,_,_=replay(prefix); found={}; movable=np.zeros((H,W),bool)
    for y in range(0,H,STEP):
        for x in range(0,W,STEP):
            g,_,go=replay(prefix+[{'x':x,'y':y}])
            if g is not None and not go and not np.array_equal(base,g):
                found.setdefault(hashlib.md5((base!=g).tobytes()).hexdigest()[:8],{'x':x,'y':y})
                movable|=(base!=g)
    return list(found.values()), movable, base

def make_h(goal_cells, movable_cells):
    """h(grid)= sum over goal cells of min-dist to a movable cell whose value != background there.
    Lower = pieces closer to goals. If no goals detected, fall back to #movable-nonbg (weak)."""
    gc=np.argwhere(goal_cells)
    mv=np.argwhere(movable_cells)
    if len(gc)==0 or len(mv)==0:
        return lambda g: 0.0
    def h(g):
        # piece cells = movable positions that are currently non-background-ish (nonzero & not frame 4)
        tot=0.0
        for (gy,gx) in gc:
            d=np.abs(mv[:,0]-gy)+np.abs(mv[:,1]-gx)
            tot+=d.min()
        return float(tot)
    return h

def best_first_levelup(prefix, buttons, cur_level, h):
    base,_,_=replay(prefix); seen={ghash(base)}
    ctr=0
    pq=[(h(base),0,ctr,[])]; nodes=0; best_h=h(base)
    while pq and nodes<NODES and time.time()-T0<TIME_BUDGET:
        hv,depth,_,sub=heapq.heappop(pq); nodes+=1
        for i,b in enumerate(buttons):
            ns=sub+[b]
            if len(ns)>DEPTH: continue
            g,nlvl,go=replay(prefix+ns)
            if g is None or go: continue
            if nlvl>cur_level: return ns,nodes,best_h
            hh=ghash(g)
            if hh in seen: continue
            seen.add(hh)
            hg=h(g); best_h=min(best_h,hg); ctr+=1
            heapq.heappush(pq,(hg,len(ns),ctr,ns))
    return None,nodes,best_h

solution=[]; level=0; per_level=[]
for _ in range(TARGET):
    if time.time()-T0>TIME_BUDGET: print("time budget hit",flush=True); break
    btns,movable,base=discover(solution)
    # goal cells: goal-palette colors, invariant (not movable), excluding the left move-bar column
    goal_cells=np.isin(base,list(GOAL_COLORS)) & (~movable)
    goal_cells[:,0]=False
    h=make_h(goal_cells,movable)
    print(f"L{level+1}: {len(btns)} buttons; goal-cells={int(goal_cells.sum())}; movable={int(movable.sum())}; h0={h(base):.0f}",flush=True)
    sub,nodes,best_h=best_first_levelup(solution,btns,level,h)
    if sub is None:
        print(f"L{level+1}: NO level-up ({nodes} nodes, best_h reached={best_h:.0f}) — stuck",flush=True); break
    solution+=sub; _,level,_=replay(solution)
    per_level.append({"level":level,"actions":len(sub)})
    print(f"  -> solved to L{level} in {len(sub)} actions (total {len(solution)}, {nodes} nodes, {time.time()-T0:.0f}s)",flush=True)

env2=arc.make(GAME,scorecard_id=arc.open_scorecard()); env2.reset(); f=None
for a in solution: f=env2.step(_game_action(GameAction,6),data=a)
final=_levels_completed(f) if f is not None else -1
human=[17,38,31,16,41,60,26,159]
print(f"\n=== VERIFIER-ROUTED RESULT ===")
print(f"  levels solved={level} (reproduced fresh env: L{final} {'OK' if final==level else 'MISMATCH'})")
print(f"  total actions={len(solution)}; per-level={[p['actions'] for p in per_level]} vs human {human[:level]}")
json.dump({"game":GAME,"levels_solved":level,"reproduced":final,"total_actions":len(solution),
           "per_level":per_level,"solution":[(b['x'],b['y']) for b in solution]},
          open("/tmp/lp85_verifier_solution.json","w"),indent=1)
print("  saved /tmp/lp85_verifier_solution.json")
