import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    
    # Find player position (color 15)
    ys, xs = np.where(g == 15)
    if len(ys) == 0:
        return g
    
    # Player is a single cell or small cluster; find centroid
    py = int(np.mean(ys))
    px = int(np.mean(xs))
    
    # Determine movement direction based on action
    # Action mapping: 1=up, 2=down, 3=left, 4=right, 5=special, 6=click, 7=?
    dx, dy = 0, 0
    if action == 1:
        dy = -1
    elif action == 2:
        dy = 1
    elif action == 3:
        dx = -1
    elif action == 4:
        dx = 1
    elif action == 5:
        # Special action - seems to fill area with 15
        pass
    elif action == 6:
        # Click action - just decrement counter at bottom right
        x = data.get('x', 0)
        y = data.get('y', 0)
        # Find the counter position (bottom-right area)
        for c in range(g.shape[1] - 1, max(g.shape[1] - 10, 0), -1):
            if g[g.shape[0]-1][c] == 5:
                g[g.shape[0]-1][c] = 0
                break
        return g
    
    if action == 5:
        # Fill the enclosed area around player with 15
        # Find connected region of 0s or other colors that should be filled
        visited = np.zeros_like(g, dtype=bool)
        queue = [(py, px)]
        while queue:
            cy, cx = queue.pop()
            if cy < 0 or cy >= g.shape[0] or cx < 0 or cx >= g.shape[1]:
                continue
            if visited[cy][cx]:
                continue
            visited[cy][cx] = True
            if g[cy][cx] != 15 and g[cy][cx] != 2:
                g[cy][cx] = 15
            else:
                continue
            for ddy, ddx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = cy+ddy, cx+ddx
                if 0 <= ny < g.shape[0] and 0 <= nx < g.shape[1] and not visited[ny][nx]:
                    queue.append((ny, nx))
        return g
    
    if dx == 0 and dy == 0:
        return g
    
    # Move player one step
    new_py = py + dy
    new_px = px + dx
    
    if new_py < 0 or new_py >= g.shape[0] or new_px < 0 or new_px >= g.shape[1]:
        return g
    
    target_val = g[new_py][new_px]
    
    # If target is a wall (color 4), can't move
    if target_val == 4:
        return g
    
    # Clear old position(s) - set to background color 5
    mask = (g == 15)
    g[mask] = 5
    
    # Place player at new position
    g[new_py][new_px] = 15
    
    # Check for collection/interaction with objects
    # Color 2 seems to be collectible items that get absorbed
    # When moving into certain areas, the path gets filled
    
    return g

def is_level_complete(grid):
    # Level complete when all 15s are collected/filled properly
    # Based on observations, completion might involve filling specific regions
    # For now, check if there's no more 15 remaining in expected positions
    # or if specific conditions are met
    count_15 = np.sum(grid == 15)
    # The game seems to progress by collecting/filling areas
    # A simple heuristic: level complete when counter reaches 0 or specific state
    # Looking at the bottom-right corner as a potential score indicator
    if grid.shape[0] > 0 and grid.shape[1] > 0:
        bottom_right = grid[-1][-1]
        if bottom_right == 0:
            return True
    return False