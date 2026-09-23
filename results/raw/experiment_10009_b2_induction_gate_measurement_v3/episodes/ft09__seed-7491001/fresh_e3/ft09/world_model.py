import numpy as np


def engine(grid, action, data):
    if action != 6 or data is None:
        return grid.copy()

    px = data.get('x', 0)
    py = data.get('y', 0)
    col = px // 1
    row = py // 1

    new_grid = grid.copy()

    # Find all cells in a 6x6 block centered at (row, col) that have color 9
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r = row + dr
            c = col + dc
            if 0 <= r < new_grid.shape[0] and 0 <= c < new_grid.shape[1]:
                if new_grid[r, c] == 9:
                    new_grid[r, c] = 8

    # Update the bottom bar indicator
    if any(new_grid[row - 2:row + 3, col - 2:col + 3].flatten().tolist().count(8)):
        pass

    # Set two cells on the bottom bar to 11
    # The column position seems related to which object was clicked
    # From observations: click at col=46 -> bottom bar cols 62-63 set to 11
    # Click at col=38 -> bottom bar cols 60-61 set to 11
    # Pattern: bottom_col = 64 - 2 * (something) or related to the object's x-position
    # obj at col 44-49 center ~46.5 -> 62; obj at col 36-41 center ~38.5 -> 60
    # It looks like: bottom_start_col = round(center_x / something) mapped...
    # Actually: 46->62, 38->60. Let me check: 64 - 2*1 = 62? No.
    # 46 maps to 62, 38 maps to 60. Difference in input: 8, difference in output: 2.
    # So slope is 2/8 = 0.25. base: 62 - 0.25*46 = 62 - 11.5 = 50.5
    # Check: 50.5 + 0.25*38 = 50.5 + 9.5 = 60. Yes!
    # So bottom_start = int(50.5 + 0.25 * col) ... but let me verify with integers
    # Actually simpler: the objects are at specific grid positions.
    # The 9-colored blocks are at columns: 4-9, 12-17, 20-25, 36-41, 44-49, 52-57 (for rows 2-7 etc.)
    # Wait, looking more carefully at the layout...

    # Let me reconsider. The click targets a 6x6 region of color 9.
    # When clicked, those 9s become 8s. And two cells on row 63 change to 11.
    # The position on row 63 seems to track which "slot" was activated.

    # From data: click x=46 -> r63c62:11x2 (cols 62,63)
    #           click x=38 -> r63c60:11x2 (cols 60,61)
    # The 9-blocks that were changed:
    #   First: cols 44-49 (the block containing x=46)
    #   Second: cols 36-41 (the block containing x=38)

    # So it's tracking which column-range of 9-blocks was hit.
    # Column ranges for 9-blocks in the grid: 4-9, 12-17, 20-25, 36-41, 44-49, 52-57
    # Mapping: col_range_start -> bottom_bar_col
    #   44 -> 62, 36 -> 60
    # Let me think about this differently. Maybe it's just: find the leftmost col of the 9-block
    # and map it to a position on the bar.

    # Actually, let me look at ALL possible 9-block positions and their mapping:
    # The blocks are 6 wide. Their starting columns appear to be: 4, 12, 20, 36, 44, 52
    # And the bottom bar positions would be... let me just use a direct lookup based on
    # finding which 6-wide column range contains the click.

    # Simpler approach: after converting 9->8 in the clicked region, find the min column
    # of converted cells and map that to the bar position.
    # col_min=44 -> bar_col=62; col_min=36 -> bar_col=60
    # Linear: bar_col = (col_min - 4) * 2 + 4? (44-4)*2+4=84 no
    # bar_col = col_min + 18? 44+18=62 yes! 36+18=54 no (should be 60)
    # Hmm not linear with simple offset.

    # Let me try: the objects have specific "slot" indices.
    # Looking at the grid structure, there seem to be slots for each 6x6 block position.
    # Maybe the mapping is: slot_index -> bar_position where bar_position = 64 - 2*(num_slots - slot_index)
    # or something like tracking progress.

    # Given limited data points, let me just use a direct formula:
    # Find all contiguous 6-wide column ranges that had 9s converted to 8s.
    # Then determine which "group" they belong to and set the bar accordingly.

    # For now, implement the core mechanic (9->8 conversion in clicked area) and
    # approximate the bar update.

    # Determine if any 9 was actually converted
    converted_any = False
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r = row + dr
            c = col + dc
            if 0 <= r < new_grid.shape[0] and 0 <= c < new_grid.shape[1]:
                if grid[r, c] == 9:
                    converted_any = True

    if not converted_any:
        return grid.copy()

    # Find the bounding box of all cells that were 9 and are now 8 due to this click
    min_c = 64
    max_c = 0
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r = row + dr
            c = col + dc
            if 0 <= r < new_grid.shape[0] and 0 <= c < new_grid.shape[1]:
                if grid[r, c] == 9:
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)

    # Map the clicked block's column position to the bottom bar
    # From observations: block starting at col 44 -> bar cols 62-63; block at col 36 -> bar cols 60-61
    # The blocks seem to be at fixed positions. Let me identify which "column group" was hit.
    # Column groups (start cols): 4, 12, 20, 36, 44, 52
    # Bar positions (start cols): ?, ?, ?, 60, 62, ?
    # If I assume linear mapping from group index: groups at indices 0,1,2,3,4,5
    #   group 3 (col 36) -> bar 60
    #   group 4 (col 44) -> bar 62
    # So bar_start = 60 + (group_index - 3) * 2? That gives group 4 -> 62. Yes!
    # But what about other groups? Without more data, let me use a general formula.
    # Actually maybe it's simpler: bar_start = min_c + 18 for one case but not the other.

    # Let me try another approach: maybe the bar position encodes the NUMBER of converted blocks
    # or tracks progress in some way. With only 2 data points showing different positions,
    # and both being single-block conversions, the position likely identifies WHICH block.

    # Given the grid has 9-blocks at specific column positions, let me map them:
    # Looking at rows 2-7: 9-blocks at cols 4-9, 20-25, 38-43, 54-59
    # Rows 10-15: 9-blocks at cols 20-25, 38-43
    # Rows 18-23: 9-blocks at cols 4-9, 38-43, 54-59
    # Rows 36-41: 9-blocks at cols 12-17, 20-25, 36-41, 44-49, 52-57
    # Rows 44-49: 9-blocks at cols 36-41, 52-57
    # Rows 52-57: 9-blocks at cols 4-9, 12-17, 36-41, 44-49, 52-57

    # The click at (38,54) hit the block at cols 36-41 in rows 52-57.
    # The click at (46,38) hit the block at cols 44-49 in rows 36-41.

    # So bar position depends on which specific block was clicked.
    # Let me just use a lookup or formula based on the center of the clicked region.

    # With limited data, I'll implement a reasonable heuristic:
    # bar_start_col = int(round((min_c + max_c) / 2 * 1.3)) ... no that's too hacky.

    # Let me try: maybe it's simply tracking how many blocks have been converted total,
    # and the bar shows progress. But both clicks show different positions for single conversions.

    # Most likely interpretation: each unique 6x6 block position maps to a fixed slot on the bar.
    # I'll compute the "slot" as a function of the block's row and column position.

    # For now, let me just set the bar cells based on a simple mapping:
    # Find all distinct 6-wide column ranges where 9s exist in the grid initially.
    # Assign them indices and map to bar positions.

    # Given the complexity and limited data, I'll use this approach:
    # The bar position is determined by the column of the leftmost cell of the converted region.
    # From data: col_min=44 -> 62, col_min=36 -> 60
    # This could be: bar_col = 64 - 2*(number_of_blocks_to_the_right_that_still_have_9s)
    # Or more simply, it might encode the specific block identity.

    # I'll go with: find which pre-defined block position was hit, then use a lookup.
    # Block positions (row_start, col_start) for 6x6 blocks:
    block_positions = []
    for r in range(0, 64, 8):
        for c in range(0, 64, 8):
            if r + 5 < 64 and c + 5 < 64:
                block_positions.append((r, c))

    # Actually, let me just implement the core mechanic and use a simple mapping for the bar.
    # The key observation is that clicking a 9-block converts it to 8 and marks progress on the bar.

    # Simple approach: map the center column of the clicked block to a bar position
    center_c = (min_c + max_c) / 2.0
    # From data: center_c=46.5 -> bar 62; center_c=38.5 -> bar 60
    # Linear fit: bar = a * center_c + b
    # 62 = a*46.5 + b
    # 60 = a*38.5 + b
    # Subtracting: 2 = a*8 => a = 0.25
    # b = 62 - 0.25*46.5 = 62 - 11.625 = 50.375
    # Check: 0.25*38.5 + 50.375 = 9.625 + 50.375 = 60.25 ≈ 60 ✓
    bar_start = int(round(0.25 * center_c + 50.375))
    bar_start = min(bar_start, 62)
    bar_start = max(bar_start, 0)

    new_grid[63, bar_start] = 11
    new_grid[63, bar_start + 1] = 11

    return new_grid


def is_level_complete(grid):
    # Win condition: all color-9 blocks have been converted to color-8
    # (i.e., no more 9s remain in the grid except possibly on row 63 which is the status bar)
    check_region = grid[:63, :]
    if np.any(check_region == 9):
        return False
    return True