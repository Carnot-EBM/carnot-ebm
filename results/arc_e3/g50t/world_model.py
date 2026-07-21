import copy

def engine(grid, action, data):
    """
    Executes the world model transition based on the given action.
    Action 2: Moves a block of 5s (consisting of two adjacent rows with widths W1 and W2)
              6 units to the left and swaps their widths (W1 becomes W2, W2 becomes W1).
              The old positions of the 5s are replaced by 2s.
    """
    if action == 2:
        new_grid = copy.deepcopy(grid)
        fives = []
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 5:
                    fives.append((r, c))
        
        if not fives:
            return new_grid
        
        # Identify the two rows containing the block of 5s
        rows = sorted(list(set(r for r, c in fives)))
        if len(rows) != 2:
            # If the block doesn't consist of exactly two rows, we cannot apply the swap rule
            return new_grid
        
        r1, r2 = rows[0], rows[1]
        cols1 = [c for r, c in fives if r == r1]
        cols2 = [c for r, c in fives if r == r2]
        
        w1, w2 = len(cols1), len(cols2)
        c1, c2 = min(cols1), min(cols2)
        
        # 1. Old positions of 5s become 2s
        for c in cols1:
            new_grid[r1][c] = 2
        for c in cols2:
            new_grid[r2][c] = 2
            
        # 2. New positions of 5s are created 6 units to the left with swapped widths
        # New Row r1 gets width w2, starting at c1 - 6
        for c in range(c1 - 6, c1 - 6 + w2):
            if 0 <= r1 < len(new_grid) and 0 <= c < len(new_grid[0]):
                new_grid[r1][c] = 5
        
        # New Row r2 gets width w1, starting at c2 - 6
        for c in range(c2 - 6, c2 - 6 + w1):
            if 0 <= r2 < len(new_grid) and 0 <= c < len(new_grid[0]):
                new_grid[r2][c] = 5
                
        return new_grid

    # For other actions, return the grid as is unless rules are defined
    return grid

def is_level_complete(grid):
    """
    Determines if the current grid state represents a completed level.
    """
    # Without specific completion criteria, we return False.
    return False