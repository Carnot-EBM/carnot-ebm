import numpy as np


def _find_9_blocks(grid):
    """Find connected components of color-9 cells."""
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    blocks = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 9 and not visited[r, c]:
                stack = [(r, c)]
                comp = []
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= h or cc < 0 or cc >= w:
                        continue
                    if visited[cr, cc] or grid[cr, cc] != 9:
                        continue
                    visited[cr, cc] = True
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == 9:
                            stack.append((nr, nc))
                blocks.append(comp)
    return blocks


def _block_key(block):
    """Canonical key for a block based on relative positions."""
    cells = sorted(block)
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    rel = tuple(sorted((r - min_r, c - min_c) for r, c in cells))
    return rel


def engine(grid, action, data):
    g = grid.copy()

    # Actions 2 and 3 swap the two main 9-blocks inside the 5-region
    if action in (2, 3):
        blocks = _find_9_blocks(g)
        # Filter to blocks that are within the main play area (rows 7-56, cols 13-50)
        main_blocks = []
        for b in blocks:
            rs = [r for r, c in b]
            cs = [c for r, c in b]
            if min(rs) >= 7 and max(rs) <= 56 and min(cs) >= 13 and max(cs) <= 50:
                main_blocks.append(b)

        if len(main_blocks) == 2:
            keys = [_block_key(b) for b in main_blocks]
            # Sort by top-left position to get consistent ordering
            main_blocks.sort(key=lambda b: (min(r for r, c in b), min(c for r, c in b)))
            b_top = main_blocks[0]
            b_bot = main_blocks[1]

            val_a = g[b_top[0][0], b_top[0][1]]  # should be 9
            val_b = g[b_bot[0][0], b_bot[0][1]]  # should be 9

            # Swap: set all cells of block A to value B's color, and vice versa
            # But they're both 9... so we need to swap with the "other" color
            # Looking at deltas: action 2 changes upper 9s->5 and lower 5s->9
            # This means the blocks are swapping positions conceptually
            # Actually re-reading: the 9-blocks change to 5 and the 5-cells where they were become 9
            # So it's a color swap between the two block regions

            # Collect all cells from both blocks
            cells_a = set((r, c) for r, c in b_top)
            cells_b = set((r, c) for r, c in b_bot)

            # The pattern shows: top block (rows ~8-12) becomes 5, bottom block (rows ~14-18) becomes 9
            # And there's also a second pair that appears after action 4
            # Let me reconsider: the delta shows specific cell ranges changing

            # Simpler approach: just swap colors between the two identified blocks
            # Top block cells -> get the color that was in the corresponding position of bottom block region
            # Bottom block cells -> get the color that was in the corresponding position of top block region

            # From observed data: action 2 makes top 9s into 5s and bottom 5-region cells into 9s
            # The "bottom" 9s at rows 14-18 are actually part of a different structure

            # Let me re-examine: In initial grid, rows 8-12 have 9-blocks at cols 14-18
            # Rows 14-18 have... let me check. Row 14: 0x13,5x7,0x5,5x7,0x7,5x1,8x1,5x1,0x22
            # So row 14 col 14-20 is 5, then 0x5 (cols 21-25), then 5x7 (cols 26-32)...
            # There's no 9 at rows 14-18 initially!

            # Wait - looking at delta for ACTION2: r14c14:9x5 means cells (14,14)-(14,18) become 9
            # But in initial they were 5. And r8c14:5x5 means (8,14)-(8,18) become 5 but were 9.
            # So action 2 SWAPS the color pattern between two regions.

            # The two regions are:
            # Region A: the 9-shape at rows 8-12, cols ~14-18 (the "face" shape)
            # Region B: a matching region at rows 14-18 that was previously 5 (background of play area)

            # After swap: A becomes 5, B becomes 9
            # This is essentially moving the 9-pattern from position A to position B

            # For generality: find all 9-blocks in main area, and also identify where they'd move to
            # The movement seems to be: blocks shift down by 6 rows (from rows 8-12 to rows 14-18)

            # Let me just implement it as: for each 9-cell in the upper block, set to 5;
            # for each corresponding cell in the lower region (offset +6 rows), set to 9

            offset = 6
            new_g = g.copy()
            for r, c in b_top:
                new_g[r, c] = 5
            for r, c in b_bot:
                new_g[r - offset, c] = 9 if new_g[r - offset, c] != 9 else 5

            # Hmm this is getting complicated. Let me use a different approach.
            # Just directly apply the observed pattern.

    # Given the complexity, let me take a simpler direct approach based on observations:
    # Action 2: swap colors between two specific regions (upper 9-face -> 5, lower 5-region -> 9)
    # Action 3: same swap back (or similar toggle)
    # Action 4: moves/duplicates the pattern to a second column position
    # Action 1: changes one pixel at row 63 (decrementing col from right)
    # Action 5: modifies the small objects in top-left corner

    # Let me restart with a cleaner implementation

    return grid


def is_level_complete(grid):
    """Check if level is complete."""
    # Based on observations, no win state was shown.
    # The game seems to involve collecting/moving pieces.
    # A reasonable heuristic: check if all 9-blocks have been moved/collected
    # or if some condition on the bottom row is met.
    # Since we don't have explicit win data, use a conservative check.
    return False