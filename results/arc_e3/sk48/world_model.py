import numpy as np

import numpy as np

def engine(grid, action, data):
    if action == 1:
        return apply_action_1(grid)
    elif action == 2:
        return apply_action_2(grid)
    elif action == 3:
        return apply_action_3(grid)
    elif action == 4:
        return apply_action_4(grid)
    elif action == 5:
        return apply_action_5(grid)
    elif action == 6:
        return apply_action_6(grid, data)
    elif action == 7:
        return apply_action_7(grid)
    return grid

def apply_action_1(grid):
    grid = grid.copy()
    # Action 1 is a vertical movement (up/down)
    # Based on observed transitions, it shifts the pattern of blocks
    # We simulate the movement by shifting rows
    # This is a simplified model based on the observed pattern
    # The exact shift amount and direction depend on the context
    # For now, we assume a simple vertical shift
    # This is a placeholder for the actual logic
    return grid

def apply_action_2(grid):
    grid = grid.copy()
    # Action 2 is a horizontal movement (left/right)
    # Similar to action 1, it shifts the pattern of blocks
    return grid

def apply_action_3(grid):
    grid = grid.copy()
    # Action 3 is a toggle or interaction
    # Based on observed transitions, it changes specific cells
    return grid

def apply_action_4(grid):
    grid = grid.copy()
    # Action 4 is a toggle or interaction
    return grid

def apply_action_5(grid):
    grid = grid.copy()
    # Action 5 is a toggle or interaction
    return grid

def apply_action_6(grid, data):
    grid = grid.copy()
    # Action 6 is a click at specific pixel coordinates
    if data and 'x' in data and 'y' in data:
        px, py = data['x'], data['y']
        # Convert pixel coordinates to logical coordinates
        logical_x, logical_y = px // 1, py // 1
        # Apply the click effect
        # This is a placeholder for the actual logic
        return grid
    return grid

def apply_action_7(grid):
    grid = grid.copy()
    # Action 7 is a toggle or interaction
    return grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on observed transitions, a win state has specific patterns
    # This is a placeholder for the actual logic
    return False

def is_level_complete(grid):
    import numpy as np
    return np.array_equal(grid, np.zeros_like(grid, dtype=int))
