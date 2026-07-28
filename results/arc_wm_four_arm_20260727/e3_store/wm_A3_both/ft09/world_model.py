import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        if grid[py, px] == 1:
            grid[py, px] = 0
            # Find the connected component of 0s connected to the clicked cell
            # and change them to 1s
            visited = set()
            stack = [(py, px)]
            while stack:
                r, c = stack.pop()
                if (r, c) in visited:
                    continue
                visited.add((r, c))
                if grid[r, c] != 0:
                    continue
                if r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1]:
                    continue
                stack.append((r + 1, c))
                stack.append((r - 1, c))
                stack.append((r, c + 1))
                stack.append((r, c - 1))
            # Change all visited 0s to 1s
            for r, c in visited:
                grid[r, c] = 1
    return grid

def is_level_complete(grid):
    # Check if the grid is complete by verifying if all 0s have been filled
    # and the grid is in a stable state
    if np.any(grid == 0):
        return False
    return True