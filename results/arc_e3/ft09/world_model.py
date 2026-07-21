def engine(grid, action, data):
    """
    Executes the world model transition based on the action and data.
    """
    if action == 6:
        x = data.get('x', 0)
        y = data.get('y', 0)
        
        # Ensure coordinates are within grid bounds
        if y < 0 or y >= len(grid) or x < 0 or x >= len(grid[0]):
            return grid
            
        # Rule for color 9 (Maroon) -> 8 (Azure)
        # Shape: 6 cells in row y, 2 cells in row y+1
        if grid[y][x] == 9:
            # Row y: x to x+5
            for col in range(x, min(x + 6, len(grid[0]))):
                if grid[y][col] == 9:
                    grid[y][col] = 8
            # Row y+1: x to x+1
            if y + 1 < len(grid):
                for col in range(x, min(x + 2, len(grid[0]))):
                    if grid[y + 1][col] == 9:
                        grid[y + 1][col] = 8
        
        # Rule for color 5 (Grey) -> 4 (Yellow)
        # Shape: 8 cells in row 0
        elif grid[y][x] == 5:
            for col in range(min(8, len(grid[0]))):
                if grid[0][col] == 5:
                    grid[0][col] = 4
                    
    return grid

def is_level_complete(grid):
    """
    Determines if the current grid state represents a completed level.
    """
    # Default completion condition: return False unless a specific goal is identified.
    return False