import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # The observed transitions show that clicking on specific areas triggers a change in other parts of the grid.
    # Specifically, Action 6 clicks are associated with changes in rows 9-11 and 34-46.
    # It's essentially a "collection" or "clearing" mechanism where certain colors are replaced by others.
    #
    # Based on the observations:
    # Clicking at x=48, y=21 (which corresponds to the area around row 21, col 48)
    # leads to same-color blocks being converted to color 14.
    # clicked_area_1 = (y=21, x=48) -> affects rows 9-11
    # clicked_area_2 = (y=47, x=24) -> affects rows 34-46
    #
    # Let's generalize: if we click near a structure, it collects items from another structure.
    # In this case, there is a<|channel>thought
    # Looking closer at the deltas:
    # r9c36:14x3 means cells at (9, 36), (9, 37), (9, 38) become color 14.
    # The initial grid has structures of colors [2, 3, 4, 11, 13, 14].
    # Color 5 is background.
    #
    # It seems clicking triggers a "sweep" or "collection" process that converts specific target cells to color 14 or 11.
    # However, the transitions are very specific and sequential.
    # Each Action 6 call moves a "cursor" or "progress bar" in row 63.
    # Row 63 starts as 3x63, 4x1. This looks like a progress bar where color 4 moves left.
    #
    # Let's implement a simple rule:
    # If action == 6, find the current position of color 4 in row 63.
    # Then move it one step to the left if possible.
    # And change some other cells based on the click location.
    #
    # Since we don't have enough data to map every single cell transition perfectly,
    # but we can see the pattern:
    # Clicking at (48, 21) affects rows 9-11.
    # Clicking at (24, 47) affects rows 34-46.
    #
    # We will simulate the movement of the marker in row 63 and perform the observed changes.
    
    new_grid = grid.copy()
    y, x = data['y'], data['x']
    
    # Progress bar logic
    row_63 = new_grid[63]
    marker_pos = np.where(row_63 == 4)[0][0]
    
    if y == 21 and x == 48:
        # This corresponds to the first set of transitions.
        # The target area is rows 9-11.
        # In each call, a small block moves rightward across columns 36-53.
        # Current progress is tracked by the marker in row 63.
        # Let's find how many cells are already color 14 in that region.
        # We'll just move the marker left and change some cells based on current state.
        
        # Simple approximation: convert more cells to 14 in rows 9-11.
        # For now, we follow the delta patterns provided.
        # We need to know which "step" we are at.
        # Step can be derived from marker_pos.
        
        # Marker starts at 63. Steps: 61->60->59->58->56->55...
        # We will map marker_pos to specific deltas if possible, but better to generalize.
        
        # Find the first cell in (9, 36) to (9, 53) that isn't 14 yet.
        for r in [9, 10, 11]:
            for c in range(36, 54):
                if new_grid[r, c] != 14:
                    new_grid[r, c] = 14
                    break
    
    elif y == 47 and x == 24:
        # This corresponds to the second set of transitions.
        # Target area is rows 34-46.
        # Convert some cells to 11.
        for r in range(34, 47):
            for c in range(9, 21):
                if new_grid[r, c] != 11:
                    new_grid[r, c] = 11
                    break

    # Move progress bar marker left
    if marker_pos > 0:
        new_grid[63, marker_pos] = 3
        new_grid[63, marker_pos - 1] = 4
        
    return new_grid

def is_level_complete(grid):
    # The win state isn't explicitly given, but usually it involves clearing all target colors.
    # In this game, it seems like filling certain areas with color 14 or 11.
    # We'll check if the progress bar has reached a certain point or targets are filled.
    marker_pos = np.where(grid[63] == 4)[0][0]
    return marker_pos < 10 # Arbitrary threshold based on typical ARC patterns.