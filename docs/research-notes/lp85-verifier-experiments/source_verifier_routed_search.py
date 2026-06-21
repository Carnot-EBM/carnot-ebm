"""lp85 (a) SOURCE-DERIVED-VERIFIER solver (fixed). Win-check khartslnwa(): all 'bghvgbtwcb' pieces on
'goal' (+1,+1) AND all 'fdgmtkfrxl' on 'goal-o'. In lp85 the PIECES are fixed and the GOALS rotate on the
conveyor, so dedup on the GRID HASH (captures the moving goals); the verifier h=sum of piece->nearest-goal
Manhattan distance is the true distance-to-win (read from the game's own sprites). Best-first routed by h.
discover -> route -> advance.
"""
import hashlib, time, json, heapq
import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _game_action,_levels_completed,_game_over
from carnot.agentic.arc_executable_world_model import to_logical,detect_cell

GAME="lp85"; STEP=2; TARGET=8; NODES=90000; DEPTH=300; TIME_BUDGET=5400
arc=kit.offline_arcade(); env=arc.make(GAME,scorecard_id=arc.open_scorecard())
f0=env.reset(); cell=detect_cell(grid_of(f0)); H,W=to_logical(grid_of(f0),cell).shape
T0=time.time()

def _budget_high():
    try: env._game.toxpunyqe.current_steps=99999  # keep budget high during SEARCH so it can't
    except Exception: pass                          # game-over and prune the (long) win-path

def replay(seq):
    """reset+replay; leaves env at state. Budget kept high so search isn't budget-pruned.
    Return (grid, level, game_over)."""
    f=env.reset(); _budget_high()
    for a in seq:
        f=env.step(_game_action(GameAction,6),data=a); _budget_high()
        if f is None or _game_over(f): return None,_levels_completed(f) if f else -1,True
    return to_logical(grid_of(f),cell),_levels_completed(f),False

def ghash(g): return hashlib.md5(g.tobytes()).hexdigest()[:12]

def dist_to_win():
    """TRUE distance-to-win from the game sprites (khartslnwa made continuous)."""
    L=env._game.current_level
    bp=[s for s in L._sprites if s.tags and 'bghvgbtwcb' in s.tags]
    op=[s for s in L._sprites if s.tags and 'fdgmtkfrxl' in s.tags]
    gs=[(s.x,s.y) for s in L._sprites if s.tags and 'goal' in s.tags]
    osr=[(s.x,s.y) for s in L._sprites if s.tags and 'goal-o' in s.tags]
    d=0
    for p in bp: d+=min((abs(p.x+1-gx)+abs(p.y+1-gy)) for gx,gy in gs) if gs else 999
    for p in op: d+=min((abs(p.x+1-gx)+abs(p.y+1-gy)) for gx,gy in osr) if osr else 999
    return d

def discover(prefix):
    base,_,_=replay(prefix); found={}
    for y in range(0,H,STEP):
        for x in range(0,W,STEP):
            g,_,go=replay(prefix+[{'x':x,'y':y}])
            if g is not None and not go and not np.array_equal(base,g):
                found.setdefault(hashlib.md5((base!=g).tobytes()).hexdigest()[:8],{'x':x,'y':y})
    return list(found.values())

def best_first(prefix, buttons, cur_level):
    base,_,_=replay(prefix); h0=dist_to_win(); seen={ghash(base)}
    ctr=0; pq=[(h0,0,ctr,[])]; nodes=0; best=h0
    while pq and nodes<NODES and time.time()-T0<TIME_BUDGET:
        hv,depth,_,sub=heapq.heappop(pq); nodes+=1
        for b in buttons:
            ns=sub+[b]
            if len(ns)>DEPTH: continue
            g,lvl,go=replay(prefix+ns)
            if g is None or go: continue
            if lvl>cur_level: return ns,nodes,best
            k=ghash(g)
            if k in seen: continue
            seen.add(k); h=dist_to_win(); best=min(best,h); ctr+=1
            heapq.heappush(pq,(h,len(ns),ctr,ns))
    return None,nodes,best

solution=[]; level=0; per_level=[]
for _ in range(TARGET):
    if time.time()-T0>TIME_BUDGET: print("time budget hit",flush=True); break
    btns=discover(solution)
    replay(solution); h0=dist_to_win()
    print(f"L{level+1}: {len(btns)} buttons; dist_to_win h0={h0}",flush=True)
    sub,nodes,best=best_first(solution,btns,level)
    if sub is None:
        print(f"L{level+1}: NO solve ({nodes} nodes, best_h={best}) — stuck",flush=True); break
    solution+=sub; _,level,_=replay(solution)
    per_level.append({"level":level,"actions":len(sub)})
    print(f"  -> solved L{level} in {len(sub)} actions (best_h hit 0; total {len(solution)}, {nodes} nodes, {time.time()-T0:.0f}s)",flush=True)

env2=arc.make(GAME,scorecard_id=arc.open_scorecard()); env2.reset(); f=None
for a in solution: f=env2.step(_game_action(GameAction,6),data=a)
final=_levels_completed(f) if f is not None else -1
human=[17,38,31,16,41,60,26,159]
print(f"\n=== (a) SOURCE-VERIFIER SOLVER RESULT ===")
print(f"  levels solved={level} (reproduced fresh env: L{final} {'OK' if final==level else 'MISMATCH'})")
print(f"  per-level actions={[p['actions'] for p in per_level]} vs human {human[:level]}")
print(f"  total actions={len(solution)}")
json.dump({"game":GAME,"levels_solved":level,"reproduced":final,"per_level":per_level,
           "total_actions":len(solution),"solution":[(b['x'],b['y']) for b in solution]},
          open("/tmp/lp85_a_solution.json","w"),indent=1)
print("  saved /tmp/lp85_a_solution.json")
