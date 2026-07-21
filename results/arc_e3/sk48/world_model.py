def engine(grid, action, data):
    """
    Executes the world model's transition logic based on the given action.
    """
    import copy
    new_grid = copy.deepcopy(grid)
    rows = len(grid)
    cols = len(grid[0])

    if action == 1:
        # Action 1: Change cells of value 3 or 5 in rows 18 and 19.
        # Row 18: All 3s and 5s become 6.
        # Row 19: 3s and 5s become 6 if column is odd, 0 if column is even.
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] in [3, 5]:
                    if r == 18:
                        new_grid[r][c] = 6
                    elif r == 19:
                        new_grid[r][c] = 6 if c % 2 != 0 else 0

    elif action == 4:
        # Action 4: Complex transformation involving values 4, 2, and 8.
        # First, find the reference column X (first cell with value 4 in row 20).
        X = -1
        if rows > 20:
            for c in range(cols):
                if grid[20][c] == 4:
                    X = c
                    break
        
        # Rule 1: General value replacements (applied before 4 -> 2/1 to avoid overlap).
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    new_grid[r][c] = 3
                elif grid[r][c] == 8:
                    new_grid[r][c] = 0
        
        # Rule 2: Pattern-based replacement for value 4 in rows 20 and 21.
        if X != -1:
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 4:
                        if r == 20:
                            # Pattern: 2, 1, 1, 2, 1, 1...
                            new_grid[r][c] = 2 if (c - X) % 3 == 0 else 1
                        elif r == 21:
                            # Pattern: 1, 1, 2, 1, 1, 2...
                            new_grid[r][c] = 2 if (c - X) % 3 == 2 else 1

    return new_grid

def is_level_complete(grid):
    """
    Determines if the current grid state represents a completed level.
    """
    # Default implementation as no specific completion criteria were provided.
    return False