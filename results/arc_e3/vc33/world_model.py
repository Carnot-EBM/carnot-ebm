def engine(grid, action, data):
    """
    Executes the world model transition based on the given action and data.
    Action 6 performs a conditional color swap based on the color of the cell 
    at the coordinates provided in the data.
    """
    if action == 6:
        # The trigger for the color swap is the color of the cell at (y, x)
        y, x = data['y'], data['x']
        trigger_color = grid[y][x]
        
        # Define the mapping based on the trigger color
        # If trigger is 7, swap 7 -> 4 and 0 -> 3
        # If trigger is 4, swap 4 -> 7 and 3 -> 0
        mapping = {}
        if trigger_color == 7:
            mapping = {7: 4, 0: 3}
        elif trigger_color == 4:
            mapping = {4: 7, 3: 0}
        
        # Apply the mapping globally across the grid
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] in mapping:
                    grid[r][c] = mapping[grid[r][c]]
                    
    return grid

def is_level_complete(grid):
    """
    Determines if the current grid state represents a completed level.
    Since the completion condition is not specified, we return False.
    """
    return False