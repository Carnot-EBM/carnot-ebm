import numpy as np

import numpy as np

def engine(grid, action, data):
    if action == 1:
        # Action 1: Toggle specific cells based on pattern
        # Based on observed transitions, Action 1 toggles cells to 0 or 9 or 12
        # The pattern seems to be filling specific regions
        # Since the observed data shows specific runs being set, we simulate that
        # However, without a clear rule, we assume it toggles based on some condition
        # Given the complexity, we'll implement a simple toggle for demonstration
        # In reality, this would need to be derived from the specific game rules
        # For now, we'll just return the grid as is, assuming no change for this action
        return grid.copy()
    elif action == 2:
        # Action 2: Similar to Action 1, but different pattern
        return grid.copy()
    elif action == 3:
        # Action 3: Similar to Action 1, but different pattern
        return grid.copy()
    elif action == 4:
        # Action 4: Similar to Action 1, but different pattern
        return grid.copy()
    elif action == 5:
        # Action 5: Similar to Action 1, but different pattern
        return grid.copy()
    elif action == 6:
        # Action 6: Click action with pixel data
        if data and 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            # Convert pixel to logical
            lx, ly = px // 1, py // 1
            if 0 <= ly < grid.shape[0] and 0 <= lx < grid.shape[1]:
                grid[ly, lx] = 0
        return grid.copy()
    elif action == 7:
        # Action 7: Similar to Action 1, but different pattern
        return grid.copy()
    else:
        return grid.copy()

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on the observed data, a win state might be when certain conditions are met
    # For now, we'll assume it's when the grid is all zeros or some other condition
    # This is a placeholder and needs to be adjusted based on the actual game rules
    return np.all(grid == 0)

def is_level_complete(grid):
    import numpy as np
    grid = np.array(grid)
    return grid[0, 0] == 1 and grid[0, 1] == 1 and grid[1, 0] == 1 and grid[1, 1] == 1
