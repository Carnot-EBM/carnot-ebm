import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, it seems like a puzzle where certain colors are 
    # shifted or toggled based on actions.
    # The provided transitions show that ACTION2 and ACTION4 shift patterns of blocks.
    # Action 2 shifts things to the right/downwards in some way.
    # Action 4 shifts things to the left/right.
    # Action 6 is a click.
    
    # Since we don'tt have enough information to actually simulate the physics,
    # and we cans't induce a general rule from the very limited data,
    # we provide a placeholder that returns the same grid if no action is 
    # # an even more basic approach:
    # In the laout, therethought be "player" cells (color 1) which might move.
    #<|channel>thought be "//no_think" //no_think own logic.
    
    # Let's try to implement a simple movement for color 1.
    # return grid.copy()
    
    # We will use the deltas directly as they are too specific.
    # But since this is a world model, we must generalize.
    # Looking at the observed transitions:
    # r63c61:1x1 -> r63c60:1x1 -> r63c59:1x1 -> r63c58:1x1
    # This suggests ACTION2 moves a cell of color 1 at row 63 leftwards.
    # And the blocks of colors 5, 9, 2, 8 are being shifted or toggled.
    
    # The patterns in the deltas show that ACTION2 shifts things rightward and then downward.
    # The pattern of 5x5 and 2x5 blocks is moving across columns 14-38.
    # The pattern of 9x5 and 5x5 blocks is moving down rows 14-54.
    
    # It seems like there's a "cursor" or "active area" that changes based on actions.
    # Let's implement the movement of color 1 (the player) first.
    # If action == 2, move color 1 left.
    # If action == 4, it doesn't seem to change color 1.
    # If action == 6, click.
    
    # next_grid = grid.copy()
    # if action == 2:
    #     # Find color 1 cells
    #     player_cells = np.argwhere(grid == 1)
    #     for r, c in player_cells:
    #         next_grid[r, c] = 0 # This is not quite correct as we are not sure what happens to the rest
    #         if c > 0:
    #             next_grid[r, c-1] = 1
    #     return next_grid
    
    # Given the complexity and limited data, returning the same grid is safest unless a clear rule exists.
    # However, for ARC, usually there's a simple logic.
    # The most consistent thing is ACTION2 moving color 1 at row 63 left.
    
    # Let's try this:
    # return grid.copy()
    
    # Since I must provide an executable world model, I will implement the movement of color 1.
    # return grid.copy()
    
    # Re-evaluating: Action 2 moves color 1 from (63, 61) -> (63, 60) -> (63, 59) -> (63, 58).
    # That's exactly 4 times action 2 was called.
    # So action 2 = move player left.
    
    # For the blocks: they seem to be "filled" or "cleared".
    # In each ACTION2 call, a new block of colors is modified.
    # This looks like a painting process.
    
    # Because we can't induce the full painting logic without more data, 
    # we'll focus on the player movement and returning the same grid otherwise.
    
    new_grid = grid.copy()
    if action == 2:
        player_cells = np.argwhere(grid == 1)
        for r, c in player_cells:
            new_grid[r, c] = 0
            if c > 0:
                new_grid[r, c - 1] = 1
    elif action == 4:
        # No observed change in player position for Action 4.
        pass
    return new_grid

def is_level_complete(grid):
    # Win state usually involves collecting all items or reaching a goal.
    # Since no win state was provided, return False.
    return False