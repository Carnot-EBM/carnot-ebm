import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (8x8 int). Return predicted next grid.
    # ACTION1 is a click action that toggles the cell at (data['y'], data['x']).
    # The initial grid is all zeros. The first action is a click.
    # Since the initial grid is empty and no other rules are observed,
    # the only effect is toggling the clicked cell to 1.
    if action == 1:
        # Convert pixel coordinates to logical coordinates
        py = data['y']
        px = data['x']
        ly = py // 8
        lx = px // 8
        # Toggle the cell
        grid[ly, lx] = 1
    return grid

def is_level_complete(grid):
    # Since the game starts with an empty grid and the first action is a click,
    # and no win condition is observed, we assume the level is never complete
    # in this specific scenario.
    return False