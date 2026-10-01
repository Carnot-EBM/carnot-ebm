import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    
    # Track the "progress" marker on column 63
    # Find the highest row that has value 5 in col 63
    col63 = g[:, 63]
    rows_with_5 = np.where(col63 == 5)[0]
    if len(rows_with_5) > 0:
        current_row = int(np.max(rows_with_5))
    else:
        current_row = -1
    
    # Actions 1-5 increment progress (move marker down by 1)
    if action in [1, 2, 3, 4, 5]:
        next_row = current_row + 1
        if next_row < 64:
            g[next_row, 63] = 5
    
    # Actions 3 and 7 swap colors 5 and 4 in the play area (cols 0-62)
    if action in [3, 7]:
        mask_5 = (g == 5) & (np.arange(g.shape[1])[None, :] < 63)
        mask_4 = (g == 4) & (np.arange(g.shape[1])[None, :] < 63)
        
        temp = g[mask_5].copy()
        g[mask_5] = 4
        g[mask_4] = 5
        
        # Restore any cells that were originally 5 but are now 4? 
        # Wait, simple swap logic:
        # Original 5 -> 4
        # Original 4 -> 5
        # This is exactly what the above does.
        
    return g

def is_level_complete(grid):
    # Win state likely when all 5s and 4s are swapped to a specific configuration or progress reaches end.
    # Given no explicit win grid, assume completion when progress marker reaches row 63 (bottom).
    # Or perhaps when all "puzzle" pieces are solved.
    # Looking at the data, actions just move the marker and swap colors.
    # A common ARC pattern: complete when the board matches a target or a counter hits max.
    # Let's check if col 63 has a 5 at the very bottom (row 63).
    if grid[63, 63] == 5:
        return True
    
    # Alternative: Maybe it's about the number of swaps? No, stateless engine.
    # Let's guess based on typical "fill the bar" mechanics.
    # If the bar (col 63) is full? It only holds one 5 at a time in observations.
    
    # Re-evaluating: The prompt says "induce... win state". 
    # Without a WIN STATE grid provided, I must infer.
    # Often, these games complete when the player interacts with a goal object.
    # Here, the "goal" might be related to the 11s or the specific arrangement.
    # However, the most distinct changing feature is the column 63 marker.
    # If the marker reaches the end, maybe that's it.
    
    # Another possibility: The game is complete when all 0s are gone? Or all 5/4 swapped correctly?
    # Since I don't have the target, I will default to False unless a clear condition is met.
    # But wait, looking at the objects: obj12 is color 5 at row 63. 
    # In INITIAL GRID: r63:5x63,11x1. So row 63 cols 0-62 are 5. Col 63 is 11.
    # The marker moves DOWN col 63.
    # If the marker hits row 63, it would overwrite the 11? Or stop before?
    # Let's assume completion when the marker has traversed the board, i.e., reached row 63.
    
    return grid[63, 63] == 5