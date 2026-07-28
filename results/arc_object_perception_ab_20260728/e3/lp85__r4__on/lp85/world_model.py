import numpy as np

import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 0:
        if data is None:
            # Move Up
            for r in range(H - 1, -1, -1):
                for c in range(W):
                    if new_grid[r, c] == 14:
                        for dr in range(1, H):
                            if r + dr < H and new_grid[r + dr, c] == 3:
                                new_grid[r + dr, c] = 5
                                break
                        break
        else:
            # Move Down
            for r in range(H):
                for c in range(W):
                    if new_grid[r, c] == 14:
                        for dr in range(1, H):
                            if r + dr < H and new_grid[r + dr, c] == 3:
                                new_grid[r + dr, c] = 5
                                break
                        break
    elif action == 1:
        if data is None:
            # Move Left
            for r in range(H):
                for c in range(W - 1, -1, -1):
                    if new_grid[r, c] == 14:
                        for dc in range(1, W):
                            if c - dc >= 0 and new_grid[r, c - dc] == 3:
                                new_grid[r, c - dc] = 5
                                break
                        break
        else:
            # Move Right
            for r in range(H):
                for c in range(W):
                    if new_grid[r, c] == 14:
                        for dc in range(1, W):
                            if c + dc < W and new_grid[r, c + dc] == 3:
                                new_grid[r, c + dc] = 5
                                break
                        break
    elif action == 2:
        if data is None:
            # Move Up-Left
            for r in range(H - 1, -1, -1):
                for c in range(W - 1, -1, -1):
                    if new_grid[r, c] == 14:
                        for dr in range(1, H):
                            for dc in range(1, W):
                                if r + dr < H and c - dc >= 0 and new_grid[r + dr, c - dc] == 3:
                                    new_grid[r + dr, c - dc] = 5
                                    break
                            if new_grid[r + dr, c - dc] == 5:
                                break
                        break
        else:
            # Move Down-Right
            for r in range(H):
                for c in range(W):
                    if new_grid[r, c] == 14:
                        for dr in range(1, H):
                            for dc in range(1, W):
                                if r + dr < H and c + dc < W and new_grid[r + dr, c + dc] == 3:
                                    new_grid[r + dr, c + dc] = 5
                                    break
                            if new_grid[r + dr, c + dc] == 5:
                                break
                        break
    elif action == 3:
        if data is None:
            # Move Up-Right
            for r in range(H - 1, -1, -1):
                for c in range(W):
                    if new_grid[r, c] == 14:
                        for dr in range(1, H):
                            for dc in range(1, W):
                                if r + dr < H and c + dc < W and new_grid[r + dr, c + dc] == 3:
                                    new_grid[r + dr, c + dc] = 5
                                    break
                            if new_grid[r + dr, c + dc] == 5:
                                break
                        break
        else:
            # Move Down-Left
            for r in range(H):
                for c in range(W):
                    if new_grid[r, c] == 14:
                        for dr in range(1, H):
                            for dc in range(1, W):
                                if r + dr < H and c - dc >= 0 and new_grid[r + dr, c - dc] == 3:
                                    new_grid[r + dr, c - dc] = 5
                                    break
                            if new_grid[r + dr, c - dc] == 5:
                                break
                        break
    elif action == 4:
        if data is None:
            # Move Up-Left
            for r in range(H - 1, -1, -1):
                for c in range(W - 1, -1, -1):
                    if new_grid[r, c] == 14:
                        for dr in range(1, H):
                            for dc in range(1, W):
                                if r + dr < H and c - dc >= 0 and new_grid[r + dr, c - dc] == 3:
                                    new_grid[r + dr, c - dc] = 5
                                    break
                            if new_grid[r + dr, c - dc] == 5:
                                break
                        break
        else:
            # Move Down-Right
            for r in range(H):
                for c in range(W):
                    if new_grid[r, c] == 14:
                        for dr in range(1, H):
                            for dc in range(1, W):
                                if r + dr < H and c + dc < W and new_grid[r + dr, c + dc] == 3:
                                    new_grid[r + dr, c + dc] = 5
                                    break
                            if new_grid[r + dr, c + dc] == 5:
                                break
                        break
    elif action == 5:
        if data is None:
            # Move Up-Right
            for r in range(H - 1, -1, -1):
                for c in range(W):
                    if new_grid[r, c] == 14:
                        for dr in range(1, H):
                            for dc in range(1, W):
                                if r + dr < H and c + dc < W and new_grid[r + dr, c + dc] == 3:
                                    new_grid[r + dr, c + dc] = 5
                                    break
                            if new_grid[r + dr, c + dc] == 5:
                                break
                        break
        else:
            # Move Down-Left
            for r in range(H):
                for c in range(W):
                    if new_grid[r, c] == 14:
                        for dr in range(1, H):
                            for dc in range(1, W):
                                if r + dr < H and c - dc >= 0 and new_grid[r + dr, c - dc] == 3:
                                    new_grid[r + dr, c - dc] = 5
                                    break
                            if new_grid[r + dr, c - dc] == 5:
                                break
                        break
    elif action == 6:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 5
    elif action == 7:
        if data is None:
            # Move Up-Left
            for r in range(H - 1, -1, -1):
                for c in range(W - 1, -1, -1):
                    if new_grid[r, c] == 14:
                        for dr in range(1, H):
                            for dc in range(1, W):
                                if r + dr < H and c - dc >= 0 and new_grid[r + dr, c - dc] == 3:
                                    new_grid[r + dr, c - dc] = 5
                                    break
                            if new_grid[r + dr, c - dc] == 5:
                                break
                        break
        else:
            # Move Down-Right
            for r in range(H):
                for c in range(W):
                    if new_grid[r, c] == 14:
                        for dr in range(1, H):
                            for dc in range(1, W):
                                if r + dr < H and c + dc < W and new_grid[r + dr, c + dc] == 3:
                                    new_grid[r + dr, c + dc] = 5
                                    break
                            if new_grid[r + dr, c + dc] == 5:
                                break
                        break
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 3 and (r < 13 or c < 1):
                return False
    return True

import numpy as np

def is_level_complete(grid):
    return np.all(grid == 0)
