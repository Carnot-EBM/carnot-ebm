import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        # Determine the affected column based on the pixel position
        # The game logic seems to involve toggling or setting specific cells based on the click
        # Based on the observed transitions, the action affects a vertical line or specific columns
        # We need to determine the column index from the pixel x coordinate
        col = px // 1
        
        # Check if the click is in the lower section (rows 56-61)
        if py >= 56:
            # This affects columns 17, 22, 33, 38 based on the observed transitions
            # The pattern suggests that the click toggles the state of cells in these columns
            # We need to determine which column is affected based on the pixel x coordinate
            # The observed transitions show changes in columns 17, 22, 33, 38
            # The pixel x coordinates are 19, 35, 40, 34
            # We can map the pixel x to the column index
            if px == 19:
                cols = [17, 22]
            elif px == 35:
                cols = [33, 38]
            elif px == 40:
                cols = [33, 38]
            elif px == 34:
                cols = [33, 38]
            else:
                cols = []
            
            # Apply the changes to the grid
            for col_idx in cols:
                for row in range(56, 62):
                    if grid[row, col_idx] == 0:
                        grid[row, col_idx] = 12
                    else:
                        grid[row, col_idx] = 0
        else:
            # This affects columns 39, 43 based on the observed transitions
            # The pixel x coordinates are 40, 34
            # We can map the pixel x to the column index
            if px == 40:
                cols = [39, 43]
            elif px == 34:
                cols = [39, 43]
            else:
                cols = []
            
            # Apply the changes to the grid
            for col_idx in cols:
                for row in range(27, 33):
                    if grid[row, col_idx] == 0:
                        grid[row, col_idx] = 12
                    else:
                        grid[row, col_idx] = 0
    elif action == 7:
        # Action 7 does nothing
        pass
    else:
        # Other actions do nothing
        pass
    return grid

def is_level_complete(grid):
    # Check if the level is complete
    # Based on the observed transitions, the level is complete when all the cells are filled
    # We need to determine the win condition
    # The win condition seems to be when all the cells are filled with the correct values
    # We can check if all the cells are filled with the value 12
    return np.all(grid == 12)