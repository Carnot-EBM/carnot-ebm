import numpy as np

import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where ACTION 4 (Right) moves a set of blocks/objects.
    # Based on the transitions, ACTION 4 shifts a pattern of cells (likely '0's and other colors)
    # horizontally by 1 unit.
    # Let's identify the objects being moved.
    # In each transition, the same relative structure of cells is moving right.
    # # We observe that r0c16 becomes 0x1, then r0c17, etc. This suggests a "cursor" or "player" at r0.
    # Looking at theC delta, we<|channel>thought
    # The movement is a shift of all non-background color (where background is 10)
    # but only for certain rows/regions.
    # Specifically, it looks like everything not equal to 10 in the range row 14 to 31
    # and column 11 to 47 is shifting right.
    
    new_grid = grid.copy()
    if action == 4:
        # Identify the region that moves.
        # It appears to be a complex shape consisting of several columns of blocks.
        # We see changes starting from col 11 and ending around col 47.
        # The observed deltas show a specific set of cells changing.
        # Shift everything in the area [14:32, 11:48] one pixel to the right.
        # Note: the cursor at r0 also moves.
        # Let's find the current position of the '0' at r0.
        # In the initial grid, r0 has 10x16, 4x32, 10x16. No '0'.
        # After first ACTION 4, r0c16 becomes 0.
        # Then r0c17, then r0c18...
        # This suggests the player/cursor starts "off-screen" or at index 15 and moves into view.
        
        # Find the cursor (value 0) at row 0.
        cursor_col = -1
        for c in range(grid.shape[1]):
            if grid[0, c] == 0:
                cursor_col = c
                break
        
        # Move cursor
        if cursor_col == -1:
            # First move: place it at col 16
            new_grid[0, 16] = 0
        else:
            # Shift existing cursor
            new_grid[0, cursor_col] = 10 # Reset old pos
            if cursor_col + 1 < grid.shape[1]:
                new_grid[0, cursor_col + 1] = 0

        # Now shift the blocks in the main area.
        # The observed deltas show a specific pattern of cells shifting right.
        # We identify the 'active' region as rows 14 to 31.
        # For these rows, we find all non-10 values and shift them.
        # Let's refine this based on the delta: r14c11 becomes 10x3, r14c26 becomes 0x3...
        # This means the gap (0) is moving right, and the block (10) is filling in behind.
        # Wait, looking closer at "r14c11:10x3", if the original was 0x15, then 10x3 replaces part of that 0.
        # So it's not just shifting; it's like a sliding window or a set of objects moving.
        
        # Simple rule: Shift everything in row 14-31 from col 11 to 47 one unit right.
        for r in range(14, 32):
            row_segment = grid[r, 11:48]
            shifted_segment = np.roll(row_segment, 1)
            # The value coming in from the left should be background (10).
            shifted_segment[0] = 10
            new_grid[r, 11:48] = shifted_segment

    return new_grid

def is_level_complete(grid):
    # No win state provided, but typically it involves reaching a goal or clearing blocks.
    # Since we don't have a target, return False unless a specific condition is met.
    # In many ARC games, completing means all cells of a certain color are gone or aligned.
    return False

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition is that all cells of the same color (excluding background 0)
    must be connected (4-connectivity) and form a single contiguous block.
    """
    grid = np.array(grid)
    colors = np.unique(grid)
    colors = colors[colors != 0]
    
    if len(colors) == 0:
        return False

    for color in colors:
        # Find all cells of this color
        cells = np.argwhere(grid == color)
        if len(cells) == 0:
            continue
            
        # BFS to check connectivity
        start_node = tuple(cells[0])
        visited = {start_node}
        queue = [start_node]
        
        while queue:
            current = queue.pop(0)
            r, c = current
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = (r + dr, c + dc)
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr, nc] == color and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        if len(visited) != len(cells):
            return False
            
    return True
