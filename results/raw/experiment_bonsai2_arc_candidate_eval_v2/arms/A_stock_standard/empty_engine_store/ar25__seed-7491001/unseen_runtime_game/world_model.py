import numpy as np

def engine(grid, action, data):
    new_grid = grid.copy()
    H, W = new_grid.shape
    
    # Track progress on column 63
    col63_5_count = int(np.sum(new_grid[:, 63] == 5))
    
    if action in [1, 2, 3, 4]:
        # Determine direction based on action
        # Action 1: Up, Action 2: Down, Action 3: Left, Action 4: Right
        
        # Find all "blocks" (connected components of colors 4 and 5)
        visited = np.zeros((H, W), dtype=bool)
        blocks = []
        
        for r in range(H):
            for c in range(W):
                if not visited[r, c] and new_grid[r, c] in [4, 5]:
                    # BFS to find connected component
                    queue = [(r, c)]
                    comp_cells = []
                    while queue:
                        cr, cc = queue.pop(0)
                        if 0 <= cr < H and 0 <= cc < W and not visited[cr, cc] and new_grid[cr, cc] in [4, 5]:
                            visited[cr, cc] = True
                            comp_cells.append((cr, cc))
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and new_grid[nr, nc] in [4, 5]:
                                    queue.append((nr, nc))
                    blocks.append(comp_cells)
        
        # Determine movement direction vector
        if action == 1:
            dr, dc = -1, 0
        elif action == 2:
            dr, dc = 1, 0
        elif action == 3:
            dr, dc = 0, -1
        else: # action == 4
            dr, dc = 0, 1
        
        # For each block, try to move it by (dr, dc)
        # A block can move if all its cells' target positions are empty (value 9 or 0? No, 9 is background, 0 is hole inside block)
        # Actually, looking at the data, the blocks move into space that was previously occupied by other parts of the same block or just empty space.
        # The "empty" space for movement seems to be color 9 (background).
        # However, the blocks contain holes (color 0). When moving, the whole shape shifts.
        
        moved_any = False
        for block in blocks:
            new_positions = []
            valid_move = True
            for r, c in block:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    valid_move = False
                    break
                # Check if target cell is part of this block (it will be vacated) or is background (9)
                # If it's another block or wall (10, 11), cannot move.
                # Note: The grid has walls (10, 11) and background (9).
                # The blocks themselves are 4/5 with 0 holes.
                
                # We need to check against the ORIGINAL grid state before any moves? 
                # Or sequential? Given they don't overlap, checking against original grid is safe if we assume no collisions between blocks.
                # But wait, do blocks collide? In the examples, they seem to move independently into empty space.
                
                target_val = new_grid[nr, nc]
                # Is the target cell part of THIS block?
                is_part_of_this_block = False
                for br, bc in block:
                    if br == nr and bc == nc:
                        is_part_of_this_block = True
                        break
                
                if is_part_of_this_block:
                    continue # Moving into a spot that was occupied by itself is fine (shift)
                
                # If not part of self, must be background (9) to allow movement?
                # What about moving into a hole (0) of another block? Unlikely given layout.
                # Let's assume only color 9 allows entry.
                if target_val != 9:
                    valid_move = False
                    break
            
            if valid_move:
                moved_any = True
                # Apply move: clear old positions, set new positions
                # First, record values
                vals = {}
                for r, c in block:
                    vals[(r, c)] = new_grid[r, c]
                
                # Clear old
                for r, c in block:
                    new_grid[r, c] = 9
                
                # Set new
                for r, c in block:
                    nr, nc = r + dr, c + dc
                    new_grid[nr, nc] = vals[(r, c)]

        # Update progress marker on column 63
        # The marker moves down one row each time an action 1-4 is performed.
        # It starts at row 0 (initially all 11s). After first action, r0c63 becomes 5.
        # So we find the current bottom-most 5 in col 63 and put a 5 below it?
        # Or simply: count how many actions have been taken? No, stateless engine.
        # Observation: 
        # Initial: col 63 is all 11.
        # Action 4 -> r0c63=5.
        # Action 1 -> r1c63=5.
        # Action 2 -> r2c63=5.
        # ...
        # So the "progress" is just filling col 63 with 5s from top to bottom.
        
        if moved_any:
            # Find the next empty spot in col 63 starting from top
            for r in range(H):
                if new_grid[r, 63] == 11:
                    new_grid[r, 63] = 5
                    break

    elif action == 5:
        # Action 5 also advances progress but does NOT move blocks.
        # From data: ACTION5 only changes r<c>c63:5x1
        for r in range(H):
            if new_grid[r, 63] == 11:
                new_grid[r, 63] = 5
                break

    elif action == 6:
        # Click. Data shows no change in examples.
        pass

    return new_grid

def is_level_complete(grid):
    # Win state likely when all 5s are placed on column 63?
    # Or when specific condition met.
    # Given no win state grid provided, we guess based on typical ARC patterns.
    # Maybe when col 63 is full of 5s?
    # Or maybe when all blocks have reached a certain position?
    # Let's check if col 63 is entirely 5 (except bottom row which is part of obj12 color 5 anyway).
    # Actually, row 63 is already 5s. Col 63 rows 0-62 start as 11.
    # If they all become 5, that might be it.
    
    # Alternative: The game name "unseen_runtime_game" doesn't give hints.
    # Let's look at the structure. There are two main moving objects (left and right clusters).
    # Perhaps completion is when they meet or reach edges?
    
    # Without explicit win state, I will assume completion when column 63 is fully filled with 5s (rows 0 to 62).
    # Note: Row 63 col 63 is already 5 in initial state (part of bottom bar).
    # So checking rows 0..62 for value 5.
    
    if grid.shape[1] < 64:
        return False
        
    col = grid[:, 63]
    # Check if all cells in col 63 from row 0 to 62 are 5
    # Initial state has 11s there. They turn into 5s one by one.
    # If all are 5, level complete.
    
    target_region = col[:63]
    return np.all(target_region == 5)