import copy
from collections import deque
import numpy as np
from arcengine.enums import GameAction

def grid_of(frame) -> np.ndarray:
    arr = np.array(frame.frame if hasattr(frame, "frame") else frame)
    if arr.ndim == 3:
        arr = arr[-1]
    return arr.astype(np.int16)

def compute_grid_delta(prev: np.ndarray, nxt: np.ndarray) -> dict:
    if prev.shape != nxt.shape:
        return {"n_changed": -1}
    diff = prev != nxt
    return {"n_changed": int(diff.sum())}

def objects(grid: np.ndarray) -> list[tuple[int, int]]:
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[counts.argmax()])
    mask = grid != bg
    if not mask.any():
        return []
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    targets = []
    for i in range(h):
        for j in range(w):
            if mask[i, j] and not seen[i, j]:
                stack = [(i, j)]
                seen[i, j] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                cy = sum(c[0] for c in cells) // len(cells)
                cx = sum(c[1] for c in cells) // len(cells)
                targets.append((cy, cx))
    return targets

def discover_buttons(env, start_grid):
    objs = objects(start_grid)
    buttons = []
    original_game = copy.deepcopy(env._game)
    
    for cy, cx in objs:
        env._game = copy.deepcopy(original_game)
        f2 = env.step(GameAction.ACTION6, data={"x": int(cx), "y": int(cy)})
        delta = compute_grid_delta(start_grid, grid_of(f2))
        if delta["n_changed"] > 0:
            buttons.append((cy, cx))
            
    env._game = copy.deepcopy(original_game)
    return buttons

def plan_bfs(env, start_grid, buttons, start_levels, max_depth=20):
    original_game = copy.deepcopy(env._game)
    
    q = deque([(original_game, [])])
    seen = {start_grid.tobytes()}
    
    while q:
        curr_game, path = q.popleft()
        if len(path) > max_depth:
            continue
            
        for cy, cx in buttons:
            env._game = copy.deepcopy(curr_game)
            f2 = env.step(GameAction.ACTION6, data={"x": int(cx), "y": int(cy)})
            g2 = grid_of(f2)
            
            if (f2.levels_completed or 0) > start_levels:
                env._game = copy.deepcopy(original_game)
                return path + [(cy, cx)]
                
            state_bytes = g2.tobytes()
            if state_bytes not in seen:
                seen.add(state_bytes)
                q.append((copy.deepcopy(env._game), path + [(cy, cx)]))
                
    env._game = copy.deepcopy(original_game)
    return None

def attempt_solve(env, budget):
    f = env.reset()
    actions_taken = 0
    start_levels = f.levels_completed or 0
    solve_log = []
    
    while actions_taken < budget:
        curr_levels = f.levels_completed or 0
        if curr_levels > start_levels:
            # We solved at least one level!
            break
            
        grid = grid_of(f)
        buttons = discover_buttons(env, grid)
        if not buttons:
            break
            
        path = plan_bfs(env, grid, buttons, curr_levels, max_depth=20)
        if not path:
            break
            
        if actions_taken + len(path) > budget:
            break
            
        # Execute on real env
        for cy, cx in path:
            f = env.step(GameAction.ACTION6, data={"x": int(cx), "y": int(cy)})
            actions_taken += 1
            solve_log.append({"level": curr_levels, "action": "click", "y": int(cy), "x": int(cx)})
            
            if (f.levels_completed or 0) > curr_levels:
                break
                
    return f, actions_taken, solve_log
