import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    H, W = g.shape
    
    # Track player position on column 63
    player_row = None
    for r in range(H):
        if g[r, 63] == 5:
            player_row = r
            break
    
    if player_row is None:
        return g
    
    # Actions 1 and 2 move the player vertically
    if action == 1:
        new_row = player_row - 1
        if new_row >= 0:
            g[player_row, 63] = 11
            g[new_row, 63] = 5
    elif action == 2:
        new_row = player_row + 1
        if new_row < H:
            g[player_row, 63] = 11
            g[new_row, 63] = 5
    
    # Actions 3 and 7 shift the blocks horizontally
    if action == 3 or action == 7:
        direction = 1 if action == 3 else -1
        
        # Find all block cells (colors 4 and 5) excluding the bottom row and player column
        block_cells = []
        for r in range(63):
            for c in range(W):
                if g[r, c] in [4, 5]:
                    block_cells.append((r, c))
        
        # Group into connected components using BFS
        visited = set()
        components = []
        
        for cell in block_cells:
            if cell not in visited:
                component = []
                queue = [cell]
                while queue:
                    cur = queue.pop(0)
                    if cur in visited:
                        continue
                    visited.add(cur)
                    component.append(cur)
                    cr, cc = cur
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < 63 and 0 <= nc < W and (nr, nc) not in visited and g[nr, nc] in [4, 5]:
                            queue.append((nr, nc))
                components.append(component)
        
        # Shift each component by direction
        for comp in components:
            new_positions = []
            valid = True
            for r, c in comp:
                nc = c + direction
                if nc < 0 or nc >= W:
                    valid = False
                    break
                new_positions.append((r, nc))
            
            if valid:
                # Check for collisions with other blocks
                comp_set = set(comp)
                collision = False
                for r, c in new_positions:
                    if (r, c) not in comp_set and g[r, c] in [4, 5]:
                        collision = True
                        break
                
                if not collision:
                    # Store old values before clearing
                    old_values = [(g[r, c], r, c) for r, c in comp]
                    
                    # Clear old positions
                    for r, c in comp:
                        g[r, c] = 9
                    
                    # Set new positions
                    for val, r, c in old_values:
                        g[r, c] = val
    
    return g

def is_level_complete(grid):
    # Win state: all block cells (colors 4 and 5) are gone from the play area
    # The bottom row (row 63) has color 5 as part of the layout, so exclude it
    for r in range(63):
        for c in range(grid.shape[1]):
            if grid[r, c] in [4, 5]:
                return False
    return True