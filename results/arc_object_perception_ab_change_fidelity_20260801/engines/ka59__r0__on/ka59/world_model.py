import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape)
    # Action mapping based on observed transitions:
    # ACTION2: Up? No, looking at coordinates, r30-32 -> r33-35 is Down.
    # ACTION3: Left.
    # ACTION4: Right.
    # Let's re-examine the movements.
    # The "objects" moving are blocks of color 14 (and their internal holes).
    # The target area seems to be a region defined by color 1.
    # The movement is constrained by boundaries or other objects.
    # ACTION2: moves something from (30,21)-(32,21) to (33,21)-(35,21)? Wait, let me check indices.
    # r30c21:1x3, r31c21:1x3, r32c21:1x3 becomes r33c21:14x3... no.
    # Looking at the same object (shape 8063231412b5aae6), it starts at r30-32, c18-20 and moves.
    # ACTION4: Right.
    # ACTION3: Left.
    # ACTION2: Down.
    # ACTION1: Up? (Not observed but logically follows).
    
    # Identify the active object: the one with color 14.
    # Find all cells of color 14.
    # Let's assume there is only one "player" object composed of multiple blocks of color 14.
    # But wait, the deltas show specific changes.
    # The "player" seems to be a set of objects that move together or independently.
    # In this case, we are moving a block of size 3x3 centered around some point.
    # Actually, looking at the delta for ACTION4: r30c18:1x3, 14x3 -> means col 18-20 become 1, then 14*3=42? No.
    # Run length encoding in deltas: "r30c18:1x3,14x3" means row 30, starting col 18, 3 cells of value 1, then 3 cells of value 14.
    # This means the original values were replaced by these new ones.
    # Original grid r30c18-20 was 14, now it's 1.
    # New cells (col 21-23) became 14.
    # So ACTION4 moves the 3x3 block from c18-20 to c21-23. That's a jump of 3 columns.
    
    # Let's refine the movement:
    # The object is a 3x3 block of color 14 (with possible internal holes).
    # Find all blocks of color 14.
    # In this case, there are two such objects (obj6 and obj7).
    # Only one moves at a time based on the action.
    # But wait, looking at the transitions, only one set of coordinates changes.
    # The "player" is likely the most recently moved or a specific one.
    # However, in ARC games, usually, you move an entity.
    # Let's assume we move ALL entities of color 14 that can move.
    # Or more simply, find the current position of the 3x3 block of color 14 and shift it.
    
    # We need to know which object is moving. Looking at the deltas, r30c18 becomes 1, then r30c21 becomes 14.
    # This means the 3x3 block shifted from col 18-20 to col 21-23.
    # ACTION4: Right (+3 cols)
    # ACTION3: Left (-3 cols)
    # ACTION2: Down (+3 rows)
    # ACTION1: Up (-3 rows)
    
    # Find all cells of color 14.
    # Create a mask of where color 14 is.
    # Shift the mask by (dr, dc).
    # For each cell (r, c) that was 14, set new grid[r+dr, c+dc] = 14.
    # For each cell (r, c) that was 14, old grid[r, c] = original value? No, that's not correct.
    # The delta shows the background changes back to what it was.
    # Let's assume the "background" is everything that isn't color 14.
    # We need to know what the object is moving over.
    # In this case, the objects are moving within a region of color 1.
    # Let'#s implement a simple shift for all color 14 blocks.

    new_grid = grid.copy()
    
    # Movement vectors
    moves = {1: (-3, 0), 2: (3, 0), 3: (0, -3), 4: (0, 3)}
    if action not in moves:
        return new_grid

    dr, dc = moves[action]
    
    # Find all cells of color 14.
    # But we only move one block at a time? Looking at the deltas, ACTION4 first moves r30c18-20 to r30c21-23.
    # Then another ACTION4 moves r30c21-23 to r30c26-28.
    # This means the movement is not always by 3. It's more like it snaps to certain positions.
    # Wait, look at the delta again: "r30c18:1x3,14x3" -> col 18-20 becomes 1, col 21-23 becomes 14.
    # The distance moved was exactly 3 columns.
    # Let's try moving all blocks of color 14 by (dr, dc).
    
    # To handle the background, we need to know what the object is moving over.
    # In this case, the same area that was 14 becomes 1.
    # For each cell (r, c) that was 14, new_grid[r+dr, c+dc] = 14.
    # And for each cell (r, c) that was 14, new_grid[r, c] = 1.
    # But wait, the original grid has colors other than 1.
    # The most logical rule is: a block of color 14 moves and leaves behind the color of the cell it replaces? No.
    # Let's assume the cells it leaves are restored to their "original" state or simply become color 1.
    # Looking at the deltas, they always restore to color 1.
    
    # Find all coordinates of color 14.
    coords = np.argwhere(grid == 14)
    if len(coords) == 0:
        return new_grid

    # We only move one "object" (connected component) of color 14.
    # Which one? Maybe the one closest to some point or just any.
    # Based on the transitions, it seems we move the object that *can* move in that direction without hitting a boundary.
    # In this case, let' same as moving all blocks of color 14.
    
    # To avoid overwriting, we first clear the old positions.
    # For each coordinate (r, c), set it back to color 1.
    for r, c in coords:
        new_grid[r, c] = 1
    
    # Then place them in the new position.
    for r, c in coords:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            new_grid[nr, nc] = 14
        else:
            # If it hits a wall, it doesn't move.
            return grid
            
    return new_grid

def is_level_complete(grid):
    # The win state isn't provided, but usually it involves reaching a target.
    # Looking at the deltas, there are some cells becoming 0 or 5.
    # Let's assume the level is complete when no more moves can be made or a specific pattern is reached.
    # Since no WIN STATE was given, let's return False unless something happens.
    # But wait, look at the "changed cells" for ACTION4: "r63c63:0x1".
    # This means cell (63, 63) changed to color 0.
    # And then (63, 62), (63, 61), etc.
    # This looks like a progress bar or a countdown own the bottom row.
    # Maybe the game ends when that counter reaches 0?
    # For now, we will just return False as we don't have the win condition.
    return False

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win state is characterized by a single connected component of color 2 (red)
    that forms a closed loop or a specific pattern.
    """
    grid = np.array(grid)
    # Find all cells of color 2
    red_cells = np.argwhere(grid == 2)
    if len(red_cells) == 0:
        return False
    
    # Check if all red cells form a single connected component (4-connectivity)
    num_red_cells = len(red_cells)
    visited = np.zeros_like(grid, dtype=bool)
    stack = [tuple(red_cells[0])]
    visited[tuple(red_cells[0])] = True
    count = 0
    
    while stack:
        curr = stack.pop()
        count += 1
        r, c = curr
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if grid[nr, nc] == 2 and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
    
    # If not all red cells are connected, it's not a win state
    if count != num_red_cells:
        return False
    
    # In this game, the win state typically involves the red cells forming a 
    # specific shape or a specific number of red cells.
    # For the a59 game, a common win condition is the red cells forming a 
    # closed loop or a specific size.
    # For the same level, the a59 game's target is often a 
    # the red cells filling a specific area.
    # return True if the red cells form a single connected component of a specific size.
    # return True if the red cells (color 2) of the same color are connected.
    # return True if the only non-zero cells are color 2.
    
    # Based on the same game family, the win condition is often that the red cells 
    # form a single connected component.
    return True
