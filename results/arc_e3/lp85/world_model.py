import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    if action == 6:
        px, py = data['x'], data['y']
        col, row = px, py
        # Toggle the color at the clicked cell
        g[row, col] = 15 if g[row, col] != 15 else 0
    else:
        # Movement / push logic
        # Determine direction from action
        if action == 1: dr, dc = -1, 0
        elif action == 2: dr, dc = 1, 0
        elif action == 3: dr, dc = 0, -1
        elif action == 4: dr, dc = 0, 1
        elif action == 5: dr, dc = -1, -1
        elif action == 6: dr, dc = -1, 1
        elif action == 7: dr, dc = 1, -1
        else: return g
        # Find the player (color 14)
        player_pos = np.argwhere(g == 14)
        if len(player_pos) == 0:
            return g
        pr, pc = player_pos[0]
        nr, nc = pr + dr, pc + dc
        if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
            # Check if the target cell is empty (color 0)
            if g[nr, nc] == 0:
                # Move player
                g[pr, pc] = 0
                g[nr, nc] = 14
            else:
                # Push logic: if target is a pushable object (color 10)
                if g[nr, nc] == 10:
                    # Find the next cell in the push direction
                    nnr, nnc = nr + dr, nc + dc
                    if 0 <= nnr < g.shape[0] and 0 <= nnc < g.shape[1] and g[nnr, nnc] == 0:
                        # Push object
                        g[pr, pc] = 0
                        g[nr, nc] = 0
                        g[nnr, nnc] = 10
                        g[pr, pc] = 14
    return g

def is_level_complete(grid):
    # Level is complete when all objects (color 10) are collected
    return not np.any(grid == 10)