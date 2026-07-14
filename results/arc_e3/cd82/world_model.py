import numpy as np

def engine(grid, action, data):
    grid = grid.copy()
    if action == 1:
        # Action 1: Move Up
        for c in range(8):
            for r in range(7, -1, -1):
                if grid[r, c] != 0:
                    for nr in range(r - 1, -1, -1):
                        if grid[nr, c] == 0:
                            grid[nr, c] = grid[r, c]
                            grid[r, c] = 0
                            break
    elif action == 2:
        # Action 2: Move Down
        for c in range(8):
            for r in range(8):
                if grid[r, c] != 0:
                    for nr in range(r + 1, 8):
                        if grid[nr, c] == 0:
                            grid[nr, c] = grid[r, c]
                            grid[r, c] = 0
                            break
    elif action == 3:
        # Action 3: Move Left
        for r in range(8):
            for c in range(7, -1, -1):
                if grid[r, c] != 0:
                    for nc in range(c - 1, -1, -1):
                        if grid[r, nc] == 0:
                            grid[r, nc] = grid[r, c]
                            grid[r, c] = 0
                            break
    elif action == 4:
        # Action 4: Move Right
        for r in range(8):
            for c in range(8):
                if grid[r, c] != 0:
                    for nc in range(c + 1, 8):
                        if grid[r, nc] == 0:
                            grid[r, nc] = grid[r, c]
                            grid[r, c] = 0
                            break
    elif action == 5:
        # Action 5: Move Up-Left
        for c in range(8):
            for r in range(7, -1, -1):
                if grid[r, c] != 0:
                    for nr in range(r - 1, -1, -1):
                        for nc in range(c - 1, -1, -1):
                            if grid[nr, nc] == 0:
                                grid[nr, nc] = grid[r, c]
                                grid[r, c] = 0
                                break
                        if grid[r, c] == 0:
                            break
    elif action == 6:
        # Action 6: Move Up-Right
        for c in range(8):
            for r in range(7, -1, -1):
                if grid[r, c] != 0:
                    for nr in range(r - 1, -1, -1):
                        for nc in range(c + 1, 8):
                            if grid[nr, nc] == 0:
                                grid[nr, nc] = grid[r, c]
                                grid[r, c] = 0
                                break
                        if grid[r, c] == 0:
                            break
    elif action == 7:
        # Action 7: Move Down-Right
        for c in range(8):
            for r in range(8):
                if grid[r, c] != 0:
                    for nr in range(r + 1, 8):
                        for nc in range(c + 1, 8):
                            if grid[nr, nc] == 0:
                                grid[nr, nc] = grid[r, c]
                                grid[r, c] = 0
                                break
                        if grid[r, c] == 0:
                            break
    return grid

def is_level_complete(grid):
    return True