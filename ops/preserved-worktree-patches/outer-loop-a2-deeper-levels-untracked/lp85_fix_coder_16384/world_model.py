import numpy as np

def engine(grid, action, data):
    # Define the action rules
    if action == 6:
        px, py = data['x'] // 1, data['y'] // 1  # Convert pixel to logical coordinates
        # Example rule: toggle the value at the clicked position
        grid[py, px] = 10 if grid[py, px] != 10 else 1
    # Add more action rules as needed
    return grid

def is_level_complete(grid):
    # Define the win condition
    # Example condition: check if all '4's are replaced with '5's
    return np.all(grid == 5)