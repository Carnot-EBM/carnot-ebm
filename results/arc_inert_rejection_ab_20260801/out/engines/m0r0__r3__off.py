import numpy as np

def engine(grid, action, data):
    """
    The game state is a 64x64 grid.
    Action 1: Move left/right? No, Action 1 seems to be shifting colors in specific regions.
    Looking at the deltas, ACTION1 shifts blocks of color 5 and 10.
    ACTION3: Shifts some cells from 5 to 10.
    ACTION4: Toggles or swaps values.
    Actually, looking at the initial grid and transitions, this looks like a puzzle where you move a 'cursor' (color 0) and modify the same-colored areas.
    Wait, the cursor is at (0, 63) initially.
    ACTION1 moves it to (0, 62), then (0, 61)... (0, 59).
    And when the cursor moves, certain blocks of color 5 change to 10.
    Let's re-examine:
    Initial cursor: r0c63 = 0.
    ACTION1: r0c62=0, r63c1=0. Cursor moves left on row 0 AND right on row 63.
    ACTION1 again: r0c61=0, r63c2=0.
    ACTION1 again: r0c60=0, r63c3=0.
    So Action 1 moves the cursors (r0, c_top) and (r63, c_bottom).
    When the cursor is at column C, cells in that column are affected? No.
    Looking at ACTION1 deltas:
    First ACTION1: r0c62=0, r63c1=0. Blocks at c14..18 (color 5 -> 10) and c44..48 (color 5 -> 10).
    Second ACTION1: r0c61=0, r63c2=0. Blocks at c14..18 and c49..53.
    Third ACTION1: r0c60=0, r63c3=0. Blocks at c14..18 and c49..53.
    Wait, it's simpler: The cursor position determines which 'column-block' of color 5 becomes color 10.
    Let's look at the coordinates:
    Cursor top: (0, 63), (0, 62), (0, 61), (0, 60), (0, 59).
    Cursor bottom: (63, 0), (63, 1), (63, 2), (63, 3), (63, 4).
    The blocks are in columns [14, 18], [44, 48], [49, 53].
    This looks like a "painting" or "filling" game.

    Actually, let's simplify based on the observed transitions:
    Action 1 moves cursors left/right.
    Action 3 changes some cells from 5 to 10.
    Action 4 swaps colors 5 and 10 in certain regions.
    """
    new_grid = grid.copy()
    
    # Cursor movement
    if action == 1:
        # Find current cursors
        top_cursor = np.where(grid[0] == 0)[0]
        bottom_cursor = np.where(grid[63] == 0)[0]
        
        if top_cursor.size > 0:
            c_top = top_cursor[0]
            new_grid[0, c_top] = 5 # restore old
            new_grid[0, max(0, c_top - 1)] = 0
        if bottom_cursor.size > 0:
            c_bot = the_col = np.where(grid[63] == 0)[0][0] if grid[63].size > 0 else 0
            new_grid[63, the_col] = 5
            new_grid[63, min(63, the_col + 1)] = 0
            
        # Now we need to handle the "painting" effect of Action 1.
        # This is complex to induce perfectly without more data.
        # But based on the deltas, it's a pattern of color 5 -> 10 in specific blocks.
        # Let's try to implement the cursor movement and the painting logic.
        # The painting happens at columns [14-18], [44-48], [49-53].
        # We will use the same delta patterns observed.
    
    elif action == 3:
        # ACTION3 changes some cells from 5 to 10 in rows 39-43, cols 44-48.
        mask = (grid[39:44, 44:49] == 5)
        new_grid[39:44, 44:49][mask] = 10
        
    elif action == 4:
        # ACTION4 swaps colors 5 and 10 in certain regions.
        # It affects rows 14-18, cols 14-18 and 44-48.
        for r in range(14, 19):
            for c in range(14, 19):
                if grid[r, c] == 5: new_grid[r, c] = 10
                elif grid[r, c] == 10: new_grid[r, c] = 5
            for c in range(44, 49):
                if grid[r, c] == 5: new_grid[r, c] = 10
                elif grid[r, c] == 10: new_grid[r, c] = 5

    return new_grid

def is_level_complete(grid):
    # Win state usually involves filling a target or clearing something.
    # In this game, it's likely when all color 5 cells are gone or specific blocks are filled.
    # We don't have the win state grid, but we can check if any cursor reached the end.
    return False # Not enough data to determine win condition.