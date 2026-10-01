import numpy as np

import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    
    # Action mapping:
    # 1=Up, 2=Right, 3=Left, 4=Down, 5=Up-Alt/Collect, 6=Click, 7=Other
    
    if action == 6 and data is not None:
        px, py = data.get('x', 0), data.get('y', 0)
        x, y = int(px), int(py)
        
        # Check if clicking on a "hole" (color 0) inside a container structure
        # The game seems to have specific interactive zones.
        # Based on observations, clicks often trigger no change or minor changes.
        # However, looking at the delta for ACTION6, it was just r63c63:5x1 which looks like noise/timer.
        # Let's assume clicks don't move the main entity but might interact with specific cells.
        # Given the complexity and lack of clear movement from clicks in the provided deltas 
        # (mostly just the bottom-right corner changing), we treat this as a no-op for logic 
        # unless it hits a specific target. 
        
        # Wait, looking closely at the first transition:
        # ACTION6 data={'x': 31, 'y': 28} -> changed r63c63:5x1
        # This suggests the click itself didn't cause the visible object movement, 
        # or the "changed cell" is an artifact of the level timer/state update.
        # The subsequent actions (5, 2, 4) show large movements of color 15/2 objects.
        
        # Hypothesis: There is a player/entity that moves with keys.
        # Color 15 appears to be the moving entity (or part of it).
        # In Initial Grid: obj12 (color 15) is at bbox=(25, 26, 31, 37).
        # After Action 5 (Up?): The 15s moved UP into the area previously occupied by 0s?
        # Let's trace Action 5 (level 0->0):
        # Changed: r34-38 c27-36 became 15. Previously these were 0 (obj13).
        # So Action 5 moved the 15-block UP into the 0-hole.
        
        # Let's identify the "Player". It seems to be the cluster of 15s.
        # Or maybe the Player is invisible and pushes things?
        # No, the 15s are clearly moving.
        
        # Let's look for the current position of the 15-cluster.
        # We need to find the connected component of 15s.
        
    # General Movement Logic
    # Identify the "active" object. Based on deltas, color 15 moves significantly.
    # Color 2 also moves (it forms a border around the 15s in some frames?).
    
    # Let's try to detect the player as the largest connected component of color 15.
    # If no 15s exist, maybe it's something else? But initial grid has 15s.
    
    def get_largest_component(grid, target_color):
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        max_size = 0
        max_coords = []
        
        for i in range(h):
            for j in range(w):
                if grid[i, j] == target_color and not visited[i, j]:
                    # BFS
                    queue = [(i, j)]
                    visited[i, j] = True
                    comp = []
                    while queue:
                        y, x = queue.pop(0)
                        comp.append((y, x))
                        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ny, nx = y+dy, x+dx
                            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and grid[ny, nx] == target_color:
                                visited[ny, nx] = True
                                queue.append((ny, nx))
                    if len(comp) > max_size:
                        max_size = len(comp)
                        max_coords = comp
        
        return max_coords

    player_cells = get_largest_component(g, 15)
    
    if not player_cells:
        return g

    # Determine direction vector
    dy, dx = 0, 0
    if action == 1 or action == 5: # Up (Action 5 seemed to move up too? Let's check Action 2/4)
        # Action 2 moved Right? 
        # Initial 15s at rows 25-31.
        # After Action 4 (Down?): 15s appeared lower? 
        # Let's re-examine Action 4 delta:
        # r21c39... r36c43... lots of 15s appearing in a diagonal pattern moving Down-Right?
        # Actually, Action 4 delta shows 15s spreading out.
        
        # Let's look at Action 2 (Right?):
        # Delta shows 15s moving from left side to right side?
        # In the second Action 2 block, we see 15s at c39-c45 etc.
        
        # Standard ARC mapping: 1=Up, 2=Right, 3=Left, 4=Down.
        # But Action 5 also caused movement. Maybe 5 is "Jump" or "Interact"?
        # The first Action 5 moved 15s UP into the hole.
        
        if action == 1: dy, dx = -1, 0
        elif action == 2: dy, dx = 0, 1
        elif action == 3: dy, dx = 0, -1
        elif action == 4: dy, dx = 1, 0
        elif action == 5: 
            # Action 5 moved Up in the first instance.
            # Let's assume 5 is also Up or a special move. Given it filled the hole above, maybe it's "Move Up".
            dy, dx = -1, 0
    else:
        return g

    # Calculate new positions
    h, w = g.shape
    new_cells = []
    valid_move = True
    
    for y, x in player_cells:
        ny, nx = y + dy, x + dx
        if ny < 0 or ny >= h or nx < 0 or nx >= w:
            valid_move = False
            break
        # Check collision with walls (color 5? color 4?)
        # Color 5 seems to be the main background/wall.
        # Color 0 is empty space.
        # If target cell is not 0 and not 15 (player), it's a wall.
        if g[ny, nx] != 0 and g[ny, nx] != 15:
             # Can we push? Or is it blocked?
             # In ARC games, usually blocked by non-empty cells unless they are "pushable".
             # Here, let's assume blocked by anything that isn't empty (0) or self (15).
             valid_move = False
             break
        new_cells.append((ny, nx))

    if valid_move and len(new_cells) == len(player_cells):
        # Clear old positions
        for y, x in player_cells:
            g[y, x] = 0
        # Set new positions
        for y, x in new_cells:
            g[y, x] = 15
            
    return g

def is_level_complete(grid):
    # Win condition unknown from data. 
    # Often involves collecting all items of a certain color or reaching a specific spot.
    # Given no win state was shown, we default to False.
    # A common heuristic: if the player (15) has collected all "targets" (e.g., color 2 or 3?).
    # For now, return False as we cannot induce the exact win condition without a positive example.
    return False

def is_level_complete(grid):
    import numpy as np
    g = np.array(grid)
    if g.ndim != 2:
        return False
    h, w = g.shape
    if h < 2 or w < 2:
        return False
    corners = [g[0,0], g[0,w-1], g[h-1,0], g[h-1,w-1]]
    if any(c == 0 for c in corners):
        return False
    return True
