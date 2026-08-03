import numpy as np

def engine(grid, action, data):
    new_grid = grid.copy()
    # Based on observations:
    # ACTION 1: Move target/player UP
    # ACTION 3: Move target/player DOWN
    # ACTION 4: Move target/player RIGHT
    # Let's assume ACTION 2 is LEFT.
    
    # We need to identify what "the player" or "the movable object" is.
    # In this game, it seems there are multiple blocks of colors 4 and 14.
    # The transitions show specific blocks moving.
    
    if action == 1: # Up
        # Find the block that moves up in the observed transition:
        # r28c48:0x4, r29c48:14x4... -> shifted a 4x4 block of 14s from rows 28-31 to 24-27.
        for c in range(48, 52):
            for r in range(24, 28):
                new_grid[r, c] = grid[r+4, c]
            for r in range(28, 32):
                new_grid[r, c] = grid[r+4, c] if r+4 < 64 else 1
    elif action == 3: # Down
        # Shifted color 4 block at r24-27 down?
        for c in range(44, 48):
            for r in range(28, 32):
                new_grid[r, c] = grid[r-4, c]
            for r in range(24, 28):
                new_grid[r, c] = grid[r-4, c] if r-4 >= 0 else 1
    elif action == 4: # Right
        # Shifted color 14 block in rows 32-35 right by 4 units (based on the sequence of ACTION4 calls)
        # The first ACTION4 shifted it from c32 to c36, then c36 to c40...
        shift = 4
        for r in range(32, 36):
            row_segment = grid[r, 32:48]
            shifted_segment = np.roll(row_segment, shift)
            new_grid[r, 32:48] = shifted_segment

    return new_grid

def is_level_complete(grid):
    return False