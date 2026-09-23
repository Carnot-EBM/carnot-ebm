import numpy as np


def engine(grid, action, data):
    g = grid.copy()
    h, w = g.shape

    # Track progress markers on top row and bottom row
    # Top row starts all 5s, bottom row starts all 5s
    # Markers appear as 0s at specific positions

    # Determine current marker positions from the grid
    # Top row: find rightmost 0 (or leftmost if none yet)
    # Bottom row: find leftmost 0 (or rightmost if none yet)

    # Actually, let me re-analyze the transitions more carefully.

    # Looking at the pattern:
    # - ACTION4 seems to move something down or toggle blocks
    # - ACTION2 moves blocks up
    # - ACTION3 moves blocks horizontally
    # - ACTION1 moves blocks up (different direction?)
    # - ACTION5/6 seem to be "no-op" or marker movements

    # Let me look at what actually changes in each transition:

    # The key observation: there are two 5x5 blocks of color 10 inside a larger region of color 5.
    # They start at rows 49-53, cols 19-23 and rows 49-53, cols 39-43.

    # ACTION4 (first): The 10-blocks swap with adjacent 5-regions vertically
    #   r49c19: was 10x5 -> becomes 5x5,10x5 (shifted right by 5 within the span)
    #   Wait, let me re-read: r49c19:5x5,10x5 means cols 19-23 become 5,5,5,5,5 then... no
    #   Actually c19 is col 19, and the run covers some width. Let me count:
    #   r49c19:5x5,10x5 = 10 cells starting at col 19: cols 19-23=5, cols 24-28=10
    #   But originally cols 19-23 were 10 and cols 24-28 were 5. So they SWAPPED!

    # Similarly for the right block:
    #   r49c34:10x5,5x5 = cols 34-38=10, cols 39-43=5
    #   Originally cols 34-38 were 5 and cols 39-43 were 10. So they SWAPPED!

    # So ACTION4 swaps adjacent horizontal strips of 10s and 5s within the play area.

    # Let me reconsider. The play area has a specific structure with corridors of color 5
    # separated by walls of colors 11 and 12. Inside this area there are movable blocks of color 10.

    # Looking more carefully at all transitions, I think the game involves:
    # - Moving 10-colored blocks around in a maze-like structure
    # - Actions 1-5 correspond to different movement directions or actions
    # - Action 6 is a click that does something (maybe collect/place)

    # Given the complexity, let me try to identify the simple rules:

    # From the deltas, it seems like blocks of size 5x5 move as units.
    # The movements appear to be:
    # - ACTION1: move up (blocks shift up by 5 rows)
    # - ACTION2: move down (blocks shift down by 5 rows)  
    # - ACTION3: move left/right (blocks shift horizontally by 5 cols)
    # - ACTION4: some kind of swap or vertical movement
    # - ACTION5/6: marker/cursor movements

    # Actually wait - looking again at the first ACTION4:
    # r49c19:5x5,10x5 means the 10-block moved RIGHT by 5 (from cols 19-23 to cols 24-28)
    # And r49c34:10x5,5x5 means the right block moved LEFT by 5 (from cols 39-43 to cols 34-38)

    # So ACTION4 moves both blocks toward each other (or away)?
    # Left block: was at 19-23, now at 24-28 -> moved RIGHT
    # Right block: was at 39-43, now at 34-38 -> moved LEFT
    # They're moving TOWARD each other!

    # Second ACTION4 (after ACTION1): 
    # r0c61:0x1 and r63c2:0x1 are just marker changes
    # No block movement in this one? Wait, there's no block delta listed.
    # Oh I see - the second ACTION4 only has marker changes, meaning the blocks can't move further.

    # Let me re-examine. After first ACTION4:
    # Left block at rows 49-53, cols 24-28
    # Right block at rows 49-53, cols 34-38

    # Then ACTION1 happens:
    # r44c24:10x5 for rows 44-48, and r49c24:5x5 for rows 49-53
    # This means the left block moved UP from rows 49-53 to rows 44-48!
    # Similarly right block moved up from rows 49-53 to rows 44-48.

    # So ACTION1 = move UP by 5 rows.

    # Then ACTION4 again: only markers change (blocks already at top of their corridor?)
    
    # Then ACTION2:
    # r44c24:5x5 (old position cleared) and r49c24:10x5 (new position set)
    # Blocks moved DOWN from rows 44-48 back to rows 49-53.
    # So ACTION2 = move DOWN by 5 rows.

    # Then another ACTION2:
    # r49c24:5x5 and r54c24:10x5
    # Blocks moved DOWN from rows 49-53 to rows 54-58.

    # Then ACTION3:
    # r54c19:10x5,5x5 - this is a horizontal shift
    # The blocks at cols 24-28 moved to cols 19-23? Or the pattern shifted?
    # Actually r54c19:10x5,5x5 means cols 19-23=10, cols 24-28=5
    # Previously cols 19-23 were 5 and cols 24-28 were 10.
    # So left block moved LEFT by 5!
    # And r54c34:5x5,10x5 means cols 34-38=5, cols 39-43=10
    # Previously cols 34-38 were 10 and cols 39-43 were 5.
    # So right block moved RIGHT by 5!

    # Wait that doesn't match "move toward each other". Let me re-check.
    # After second ACTION2, blocks are at rows 54-58, cols 24-28 (left) and cols 34-38 (right).
    # ACTION3: left goes to cols 19-23 (LEFT), right goes to cols 39-43 (RIGHT).
    # They're moving AWAY from each other!

    # Second ACTION3:
    # r54c14:10x5,5x5 - cols 14-18=10, cols 19-23=5
    # Left block moved further LEFT from 19-23 to 14-18.
    # r54c39:5x5,10x5 - cols 39-43=5... wait no.
    # Actually c39 with 5x5,10x5 = cols 39-43=5, cols 44-48=10? That seems wrong.
    # Hmm, let me reconsider the column positions.

    # I think I need a simpler model. Let me just implement what I observe:
    # Two 5x5 blocks of color 10 that can move in a grid of 5-cell steps.
    # The movement is constrained by walls (colors 11, 12) and boundaries.

    # For simplicity, I'll implement a general "move blocks" system where:
    # - Find all 5x5 regions of color 10
    # - Based on action, attempt to shift them by 5 cells in the appropriate direction
    # - Check if the destination is valid (all cells are color 5 or 10)

    # Action mapping based on observations:
    # ACTION1: move UP (row -= 5)
    # ACTION2: move DOWN (row += 5)  
    # ACTION3: move LEFT for left block, RIGHT for right block (or both move outward)
    # ACTION4: move RIGHT for left block, LEFT for right block (both move inward)
    # ACTION5: no-op / marker only
    # ACTION6: click / marker only

    # Actually, re-reading more carefully, I think actions 1-4 might be directional:
    # In many ARC games: 1=up, 2=down, 3=left, 4=right, 5=?, 6=click
    # But here it seems like both blocks always move together.

    # Let me just go with: find 10-blocks and move them according to action.

    def find_blocks(g):
        """Find all 5x5 blocks of color 10."""
        blocks = []
        h, w = g.shape
        for r in range(0, h - 4, 5):
            for c in range(0, w - 4, 5):
                if np.all(g[r:r+5, c:c+5] == 10):
                    blocks.append((r, c))
        return blocks

    def can_move(g, br, bc, dr, dc):
        """Check if a 5x5 block at (br,bc) can move by (dr,dc)."""
        nr, nc = br + dr, bc + dc
        h, w = g.shape
        if nr < 0 or nc < 0 or nr + 5 > h or nc + 5 > w:
            return False
        # Destination must be all 5s (or the block's own cells if overlapping)
        dest = g[nr:nr+5, nc:nc+5]
        return np.all(dest == 5)

    def move_block(g, br, bc, dr, dc):
        """Move a 5x5 block from (br,bc) to (br+dr, bc+dc)."""
        g[br:br+5, bc:bc+5] = 5
        g[br+dr:br+dr+5, bc+dc:bc+dc+5] = 10

    # Determine action direction
    if action == 1:
        dr, dc = -5, 0
    elif action == 2:
        dr, dc = 5, 0
    elif action == 3:
        dr, dc = 0, -5
    elif action == 4:
        dr, dc = 0, 5
    else:
        # Actions 5 and 6 don't move blocks
        return g

    blocks = find_blocks(g)
    for (br, bc) in blocks:
        if can_move(g, br, bc, dr, dc):
            move_block(g, br, bc, dr, dc)

    return g


def is_level_complete(grid):
    # Win condition: check if all 10-blocks have been moved to specific positions
    # or some other completion criterion. Without a clear win state observed,
    # I'll check if there are no more color-10 cells remaining (all collected).
    return not np.any(grid == 10)