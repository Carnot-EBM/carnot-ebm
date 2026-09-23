import numpy as np


def _find_14_blocks(grid):
    """Find all 5x5 blocks of color 14."""
    h, w = grid.shape
    blocks = []
    for r in range(h - 4):
        for c in range(w - 4):
            if np.all(grid[r:r+5, c:c+5] == 14):
                blocks.append((r, c))
    return blocks


def _find_player(grid):
    """Find the player (color 11)."""
    positions = np.argwhere(grid == 11)
    if len(positions) > 0:
        return tuple(positions[0])
    return None


def _find_orange(grid):
    """Find orange cells (color 9)."""
    return set(map(tuple, np.argwhere(grid == 9)))


def engine(grid, action, data):
    new_grid = grid.copy()
    h, w = new_grid.shape

    # Track row 63 counter
    counter_val = int(new_grid[63, 0]) if new_grid[63, 0] != 0 else 0

    if action == 6 and data is not None:
        px, py = data['x'], data['y']
        lx, ly = px // 1, py // 1

        # Check if click hits a 14 block
        hit_block = False
        for (br, bc) in _find_14_blocks(new_grid):
            if br <= ly < br + 5 and bc <= lx < bc + 5:
                hit_block = True
                break

        if hit_block:
            # Find the specific 14 block that was clicked
            target_r, target_c = -1, -1
            for (br, bc) in _find_14_blocks(new_grid):
                if br <= ly < br + 5 and bc <= lx < bc + 5:
                    target_r, target_c = br, bc
                    break

            # Determine which column of 10 to fill based on position within grid structure
            # The 14 blocks are at columns: 13-17, 19-23, 25-29, 31-35, 37-41, 43-47, 49-53
            # Map to the 10-column positions
            col_map = {13: 31, 19: 36, 25: 42, 31: 49}
            # Actually let's figure out from data: clicking different x positions fills different columns
            # x=33 -> cols 31-35, x=51 -> cols 49-53, x=39 -> cols 36-41, x=45 -> cols 42-48
            # So it seems like clicking a 14 block fills the corresponding 10 column
            # Let me re-examine: the 14 blocks in rows 12-18 area are at specific columns
            
            # From the deltas, clicking fills a vertical strip of 10s
            # x=33,y=15 -> r12-18 c31-35 (width 5)
            # x=51,y=15 -> r12-18 c49-53 (width 5)  
            # x=39,y=15 -> r12-18 c36-41 (width 6)
            # x=45,y=15 -> r12-18 c42-48 (width 7)
            
            # The pattern: clicking a 14 block converts it to 10 and extends rightward
            # Actually looking more carefully, it seems like clicking fills from the left edge
            # of the clicked block's "slot" to some boundary

            # Simpler interpretation: each 14 block when clicked becomes 10
            # and the fill width depends on which block it is
            new_grid[target_r:target_r+5, target_c:target_c+5] = 10
        else:
            # Click on non-block area - just increment counter
            pass

    elif action == 3 or action == 4:
        # Arrow keys move the player (color 11) and push orange (color 9) blocks
        player = _find_player(new_grid)
        if player is None:
            return new_grid
        
        pr, pc = player
        dr, dc = 0, 0
        if action == 3:
            dr, dc = 0, 1  # right
        elif action == 4:
            dr, dc = 0, -1  # left

        nr, nc = pr + dr, pc + dc
        if 0 <= nr < h and 0 <= nc < w:
            target_val = new_grid[nr, nc]
            if target_val in [0, 5]:
                # Move player
                new_grid[pr, pc] = 5
                new_grid[nr, nc] = 11
            elif target_val == 9:
                # Push orange block
                pr2, pc2 = nr + dr, nc + dc
                if 0 <= pr2 < h and 0 <= pc2 < w:
                    if new_grid[pr2, pc2] in [0, 5]:
                        new_grid[pr, pc] = 5
                        new_grid[nr, nc] = 9
                        new_grid[pr2, pc2] = 11
                    else:
                        # Can't push into wall, but maybe swap?
                        pass

    elif action == 7:
        # Action 7 seems to be a special move that also moves the player
        player = _find_player(new_grid)
        if player is None:
            return new_grid
        
        pr, pc = player
        # From data, action 7 at certain points moves player right by swapping with 9s
        # It appears to be "move right" as well but with different behavior
        # Looking at the deltas more carefully, action 7 seems identical to action 3 (right)
        dr, dc = 0, 1
        nr, nc = pr + dr, pc + dc
        if 0 <= nr < h and 0 <= nc < w:
            target_val = new_grid[nr, nc]
            if target_val in [0, 5]:
                new_grid[pr, pc] = 5
                new_grid[nr, nc] = 11
            elif target_val == 9:
                pr2, pc2 = nr + dr, nc + dc
                if 0 <= pr2 < h and 0 <= pc2 < w:
                    if new_grid[pr2, pc2] in [0, 5]:
                        new_grid[pr, pc] = 5
                        new_grid[nr, nc] = 9
                        new_grid[pr2, pc2] = 11

    # Increment counter on row 63 for every action
    new_grid[63, 0] = counter_val + 1

    return new_grid


def is_level_complete(grid):
    """Check if all 14 blocks have been converted to 10."""
    return not np.any(grid[:63, :] == 14)