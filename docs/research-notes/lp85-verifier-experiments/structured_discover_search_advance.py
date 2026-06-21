"""lp85 FULL per-level solver: discover -> search -> advance, repeated each level.

Per level (offline, deterministic reset+replay):
  1. DISCOVER the active buttons at the current state (coarse systematic sweep; one click per distinct effect).
  2. SEARCH (BFS over discovered buttons, grid-hash dedup) for the shortest sub-sequence that increments
     levels_completed.
  3. ADVANCE: append the sub-sequence to the global solution, move to the next level.
This is the structured, no-LLM solver. The global solution is offline-reproducible by construction
(deterministic reset+replay). Generalizes to the button-puzzle family.
"""
import hashlib, time, json
from collections import deque
import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed, _game_over
from carnot.agentic.arc_executable_world_model import to_logical, detect_cell

GAME="lp85"; STEP=2; TARGET_LEVELS=8; SEARCH_NODES=3000; SEARCH_DEPTH=30; TIME_BUDGET=900
arc=kit.offline_arcade(); env=arc.make(GAME, scorecard_id=arc.open_scorecard())
f0=env.reset(); cell=detect_cell(grid_of(f0)); H,W=to_logical(grid_of(f0),cell).shape
T0=time.time()

def replay(seq):
    """reset+replay; return (grid or None, level, game_over)."""
    f=env.reset()
    for a in seq:
        f=env.step(_game_action(GameAction,6),data=a)
        if f is None: return None,-1,True
        if _game_over(f): return to_logical(grid_of(f),cell),_levels_completed(f),True
    return to_logical(grid_of(f),cell),_levels_completed(f),False

def ghash(g): return hashlib.md5(g.tobytes()).hexdigest()[:12]

def discover(prefix):
    base,_,_=replay(prefix)
    found={}
    for y in range(0,H,STEP):
        for x in range(0,W,STEP):
            g,_,go=replay(prefix+[{'x':x,'y':y}])
            if g is not None and not go and not np.array_equal(base,g):
                s=hashlib.md5((base!=g).tobytes()).hexdigest()[:8]
                found.setdefault(s,{'x':x,'y':y})
    return list(found.values())

def search_levelup(prefix, buttons, cur_level):
    """BFS over `buttons` from the prefix-state; return shortest subseq raising levels_completed."""
    base,lvl0,_=replay(prefix); seen={ghash(base)}
    q=deque([[]]); nodes=0
    while q and nodes<SEARCH_NODES and time.time()-T0<TIME_BUDGET:
        sub=q.popleft(); nodes+=1
        for b in buttons:
            ns=sub+[b]
            if len(ns)>SEARCH_DEPTH: continue
            g,nlvl,go=replay(prefix+ns)
            if g is None or go: continue
            if nlvl>cur_level: return ns,nodes
            h=ghash(g)
            if h not in seen: seen.add(h); q.append(ns)
    return None,nodes

solution=[]; level=0; per_level=[]
print(f"grid {H}x{W}, sweeping every {STEP}px", flush=True)
for li in range(TARGET_LEVELS):
    if time.time()-T0>TIME_BUDGET: print("time budget hit", flush=True); break
    btns=discover(solution)
    print(f"L{level+1}: discovered {len(btns)} distinct buttons at {[(b['x'],b['y']) for b in btns]}", flush=True)
    sub,nodes=search_levelup(solution, btns, level)
    if sub is None:
        print(f"L{level+1}: NO level-up found ({nodes} nodes searched over {len(btns)} buttons) — stuck", flush=True)
        break
    solution+=sub; _,level,_=replay(solution)
    seq_str=''.join({(4,32):'L',(58,32):'R'}.get((b['x'],b['y']), f"<{b['x']},{b['y']}>") for b in sub)
    per_level.append({"level":level,"actions":len(sub),"seq":seq_str})
    print(f"  -> solved to level {level} in {len(sub)} actions: {seq_str} (total {len(solution)} actions, {time.time()-T0:.0f}s)", flush=True)

# reproducibility validation: replay full solution from a FRESH env
env2=arc.make(GAME, scorecard_id=arc.open_scorecard()); env2.reset()
f=None
for a in solution: f=env2.step(_game_action(GameAction,6),data=a)
final_level=_levels_completed(f) if f is not None else -1
human=[17,38,31,16,41,60,26,159]
print(f"\n=== lp85 FULL SOLVER RESULT ===")
print(f"  levels solved: {level}  (reproduced on fresh env: level {final_level} {'OK' if final_level==level else 'MISMATCH'})")
print(f"  total actions: {len(solution)};  per-level: {per_level}")
print(f"  efficiency vs human baseline {human[:level]}: agent per-level actions {[p['actions'] for p in per_level]}")
json.dump({"game":GAME,"levels_solved":level,"reproduced_level":final_level,"total_actions":len(solution),
           "per_level":per_level,"solution":[(b['x'],b['y']) for b in solution]},
          open("/tmp/lp85_solution.json","w"),indent=1)
print(f"  solution saved to /tmp/lp85_solution.json")
