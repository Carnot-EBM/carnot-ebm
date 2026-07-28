import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        
        # Check if the player is at the target location
        if (py, px) in get_player_positions(new_grid):
            # Move the player
            new_grid[py, px] = 4  # Player color
            
            # Check if there are any blocks to move
            blocks = get_blocks_to_move(new_grid, px, py)
            if blocks:
                # Move the blocks
                for block in blocks:
                    new_grid[block['y'], block['x']] = 4  # Player color
                    new_grid[block['y'], block['x'] + 1] = 15  # Block color
                    new_grid[block['y'], block['x'] - 1] = 4  # Player color
            else:
                # No blocks to move, just change the player position
                new_grid[py, px] = 4  # Player color
                new_grid[py, px + 1] = 15  # Block color
                new_grid[py, px - 1] = 4  # Player color
        
        return new_grid
    else:
        # For other actions, return the grid unchanged
        return grid

def get_player_positions(grid):
    # Find the player positions in the grid
    player_positions = []
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x] == 4:
                player_positions.append((y, x))
    return player_positions

def get_blocks_to_move(grid, px, py):
    # Find the blocks to move
    blocks = []
    for block_y in range(grid.shape[0]):
        for block_x in range(grid.shape[1]):
            if grid[block_y, block_x] == 15:
                blocks.append({'y': block_y, 'x': block_x})
    return blocks

def is_level_complete(grid):
    # Check if the level is complete
    h, w = grid.shape
    for y in range(h):
        for x in range(w):
            if grid[y, x] != 0 and grid[y, x] != 4 and grid[y, x] != 15:
                return False
    return True