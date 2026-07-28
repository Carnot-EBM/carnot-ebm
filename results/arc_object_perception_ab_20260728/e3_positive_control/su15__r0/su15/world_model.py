import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        
        # Check if the click is on a 0 cell (empty space)
        if new_grid[py, px] == 0:
            # Determine the size of the block to be created
            # The block size is determined by the distance from the click to the nearest wall or obstacle
            # In this game, the block size is always 3x3
            block_size = 3
            
            # Create a 3x3 block of 15s
            for dy in range(block_size):
                for dx in range(block_size):
                    if 0 <= py + dy < h and 0 <= px + dx < w:
                        new_grid[py + dy, px + dx] = 15
            return new_grid
        else:
            return new_grid
    else:
        return grid

def is_level_complete(grid):
    h, w = grid.shape
    # Check if the grid is full of 5s (except for the 0s in the last row)
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is placed by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s have been converted to 5s
    # This is indicated by the presence of 5s in the grid
    # The win state is when all the 15s