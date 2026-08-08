import numpy as np

def engine(grid, action, data):
    """
    World model for the ARC-AGI-3 game 'm0r0'.
    
    Rules:
    - The grid contains blocks of color 10 (5x5 or 5x10) moving in a field of color 5.
    - Action 1: Move all color 10 blocks UP by 5 units.
    - Action 2: Move all color 10 blocks DOWN by 5 units.
    - Action 3: Move all color 10 blocks AWAY from the center (x=31.5).
    - Action 4: Move all color 10 blocks TOWARDS the center (x=31.5).
    - When blocks move, they leave color 5 behind.
    - When blocks overlap, they merge into a larger block of color 10.
    - "0" cells grow from the corners (0, 63) and (63, 0) based on certain actions.
    """
    new_grid = grid.copy()
    h, w = grid.shape
    
    # Identify all color 10 blocks
    blocks = []
    visited = np.zeros_like(grid, dtype=bool)
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 10 and not visited[r, c]:
                # Find the connected component of color 10
                comp = []
                stack = [(r, c)]
                visited[r, c] = True
                while stack:
                    curr_r, curr_c = stack.pop()
                    comp.append((curr_r, curr_c))
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 10 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                
                # Get bbox
                rs = [p[0] for p in comp]
                cs = [p[1] for p in comp]
                blocks.append({'y0': min(rs), 'x0': min(cs), 'y1': max(rs), 'x1': max(cs)})

    # Movement logic
    new_blocks = []
    for b in blocks:
        y0, x0, y1, x1 = b['y0'], b['x0'], b['y1'], b['x1']
        dy, dx = 0, 0
        
        if action == 1:
            dy = -5
        elif action == 2:
            dy = 5
        elif action == 3:
            # Move away from center x=31.5
            center_x = 31.5
            block_center_x = (x0 + x1) / 2
            dx = 5 if block_center_x > center_x else -5
        elif action == 4:
            # Move towards center x=31.5
            center_x = 31.5
            block_center_x = (x0 + x1) / 2
            dx = -5 if block_center_x > center_x else 5
            
        # Apply movement and clip to grid boundaries
        ny0, nx0 = max(0, min(h-1, y0 + dy)), max(0, min(w-1, x0 + dx))
        ny1, nx1 = max(0, min(h-1, y1 + dy)), max(0, min(w-1, x1 + dx))
        new_blocks.append({'y0': ny0, 'x0': nx0, 'y1': ny1, 'x1': nx1})

    # Clear old blocks and fill with color 5
    for b in blocks:
        new_grid[b['y0']:b['y1']+1, b['x0']:b['x1']+1] = 5
        
    # Place new blocks and merge
    for b in new_blocks:
        new_grid[b['y0']:b['y1']+1, b['x0']:b['x1']+1] = 10

    # Handle "0" cell growth (simplified based on observations)
    # Action 1 and 4 grow the 0-cells in the corners
    if action == 1 or action == 4:
        # Find current 0-cell boundaries
        # Top-right: (0, 63) grows left
        # Bottom-left: (63, 0) grows right
        # We need to find the leftmost 0 in row 0 and rightmost 0 in row 63
        for c in range(w - 1, -1, -1):
            if new_grid[0, c] == 0:
                if c > 0:
                    new_grid[0, c-1] = 0
                break
        for c in range(0, w):
            if new_grid[h-1, c] == 0:
                if c < w - 1:
                    new_grid[h-1, c+1] = 0
                break

    return new_grid

def is_level_complete(grid):
    """
    The level is complete when the merged color 10 block (5x10) 
    reaches the top edge of the color 5 area (y=9) and is centered (x=24-33).
    """
    # Find all color 10 blocks
    visited = np.zeros_like(grid, dtype=bool)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 10 and not visited[r, c]:
                comp = []
                stack = [(r, c)]
                visited[r, c] = True
                while stack:
                    curr_r, curr_c = stack.pop()
                    comp.append((curr_r, curr_c))
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 10 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                
                rs = [p[0] for p in comp]
                cs = [p[1] for p in comp]
                y0, x0, y1, x1 = min(rs), min(cs), max(rs), max(cs)
                
                # Win condition: 5x10 block at y=9, x=24-33
                if y0 == 9 and y1 == 13 and x0 == 24 and x1 == 33:
                    return True
    return False