import numpy as np

def engine(grid, action, data):
    # The observed transitions show a pattern of shifting colors in specific blocks.
    # Action 0 seems to trigger these shifts.
    # Based on the delta changes, it's a cyclic shift of values within certain regions.
    # Let' same identify the regions and define the<|channel>thought process for the logic.
    
    if action != 0:
        return grid.copy()

    new_grid = grid.copy()
    
    # Regions identified from deltas:
    # Region 1: (19-22, 12-63) - Horizontal shift
    # Region 2: (25-28, 12-63) - Horizontal shift
    # Region 3: (31-34, 12-63) - Horizontal shift
    # Region 4: (37-40, 12-63) - Horizontal shift
    # Region 5: (43-46, 12-63) - Horizontal shift
    # Region 6: (0-4, 0-0), (5-9, 0-0), (10-14, 0-0) - Vertical color change
    
    # The observed data is very limited. We induce a simple rule:
    # Action 0 rotates colors in specific blocks based on a sequence.
    # Sequence of values seen in deltas: [2, 10, 9, 15, 11, 2, 15] etc.
    # Let's define the same rotation for each block.
    
    # Define the blocks and their corresponding value rotations.
    blocks = [
        ((19, 22), (12, 63)),
        ((25, 28), (12, 63)),
        ((31, 34), (12, 63)),
        ((37, 40), (12, 63)),
        ((43, 46), (12, 63))
    ]
    
    # Value mapping for cyclic shifts (based on common values in deltas)
    # This is a simplified model as we only have ACTION0 examples.
    val_map = {
        2: 10, 10: 9, 9: 15, 15: 11, 11: 2
    }
    
    for r_start, r_end in [(0, 4), (5, 9), (10, 14)]:
        for r in range(r_start, r_end + 1):
            new_grid[r, 0] = 5
            
    for b_row, b_col in blocks:
        r1, r2 = b_row
        c1, c2 = b_col
        for r in range(r1, r2 + 1):
            for c in range(c1, c2):
                v = grid[r, c]
                new_grid[r, c] = val_map.get(v, v)

    return new_grid

def is_level_complete(grid):
    # No win state provided, but typically it's when certain colors are aligned or cleared.
    # Based on the initial grid and deltas, we assume a target pattern.
    # Since no WIN STATE was given, return False unless a specific condition is met.
    return False