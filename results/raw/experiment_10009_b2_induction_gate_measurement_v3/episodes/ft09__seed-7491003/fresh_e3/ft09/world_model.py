import numpy as np


def engine(grid, action, data):
    g = grid.copy()
    if action == 6 and data is not None:
        px, py = data['x'], data['y']
        col, row = px // 1, py // 1
        # Check if click hits a color-9 (fa18cdda66187c48) 6x6 block
        if 0 <= row < g.shape[0] and 0 <= col < g.shape[1]:
            # Find top-left of potential 6x6 block containing (row, col)
            for dr in range(6):
                for dc in range(6):
                    r0, c0 = row - dr, col - dc
                    if 0 <= r0 < g.shape[0] - 5 and 0 <= c0 < g.shape[1] - 5:
                        block = g[r0:r0+6, c0:c0+6]
                        if np.all(block == 9):
                            g[r0:r0+6, c0:c0+6] = 8
                            break
                else:
                    continue
                break
        # Update bottom bar indicator
        if 0 <= col < g.shape[1]:
            g[63, col] = 11
    return g


def is_level_complete(grid):
    return False