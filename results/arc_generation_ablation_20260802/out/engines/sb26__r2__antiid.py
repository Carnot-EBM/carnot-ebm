import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid
    
    px, py = data['x'], data['y']
    # The board seems to have regions. Let's identify them based on coordinates.
    # Region 1 (Top): y=0..7, x=17..46
    # Region 2 (Middle): y=24..35, x=17..46
    # Region 3 (Bottom): y=57..60, x=18..46
    
    # Check if click is in Bottom region (Region 3)
    # Based on observations:
    # Click at (36, 59) -> changes cells around c33-c38
    # Click at (20, 59) -> changes cells around c17-c22
    # { 'x': 20, 'y': 59 } -> r56c17...r61c17
    # { 'x': 44, 'y': 59 } -> r56c41...r61c41
    # { 'x': 35, 'y': 30 } -> r28c33...r31c33 and r56c41...r61c41
    
    # Let's map the clicks to specific blocks.
    # The bottom area has 4 colored blocks: 14(C), 15(M), 9(B), 11(Y).
    # Block 1: x=18..21, color=14
    # Block 2: x=22..25, color=15
    # Block 3: x=26..29, color=9
    # Block 4: x=30..33, color=11
    # Wait, looking at INITIAL GRID:
    # r57: 4x18, 14x4, 4x4, 15x4, 4x4, 9x4, 4x4, 11x4, 4x18
    # So colors are at: [18-21], [22-25] is gap, [26-29] is 15, [30-33] is gap, [34-37] is 9, [38-41] is gap, [42-45] is 11.
    # No, let's re-read the run length:
    # r57: 4x18 (col 0-17), 14x4 (col 18-21), 4x4 (col 22-25), 15x4 (col 26-29), 4x4 (col 30-33), 9x4 (col 34-37), 4x4 (col 38-41), 11x4 (col 42-45), 4x18 (col 46-63)
    # Colors in bottom region:
    # Color 14: col 18-21
    # Color 15: col 26-29
    # Color 9: col 34-37
    # Color 11: col 42-45
    
    # Now look at ACTION6 data={'x': 36, 'y': 59} -> changes cells around c33-c38.
    # This click is near color 9 (col 34-37).
    # Click at (20, 59) -> changes cells around c17-c22.
    # This click is near color 14 (col 18-21).
    # { 'x': 44, 'y': 59 } -> changes cells around c17-c22? No, r56c41...r61c41.
    # This click is near color 11 (col 42-45).
    # { 'x': 35, 'y': 30 } -> r28c33...r31c33 and r56c41...r61c41.
    # This click is near the middle region blocks.
    
    # Let's refine the block mapping for bottom area:
    # Block 0: x=18..21, Color 14
    # Block 1: x=26..29, Color 15
    # Block 2: x=34..37, Color 9
    # Block 3: x=42..45, Color 11
    
    # If you click a block in the bottom area, it disappears (becomes color 4) or toggles.
    # The observations show that when clicking at y=59, the block at that X position becomes 0 or 4.
    # Specifically, if we click (36, 59), the cells around c33-c38 become 0.
    # Actually, let's look at the delta: "r56c33:0x6 r57c33:0x1 r57c38:0x1 r58c33:0x1 r58c38:0x1 r59c33:0x1 r59c38:0x1 r60c33:0x1 r60c38:0x1 r61c33:0x6"
    # This is removing a vertical strip of width 6? No, col 33 to 38 is width 6.
    # It seems like clicking on a block removes it and potentially triggers something else.
    
    # Let's map clicks to blocks again based on data:
    # Click (20, 59) -> Block 0 (Color 14) removed.
    # Click (36, 59) -> Block 2 (Color 9) removed.
    # Click (44, 59) -> Block 3 (Color 11) removed.
    # Now consider Middle region (y=30):
    # { 'x': 23, 'y': 30 } -> Color 9 appears at r28-31 c21-24. Bottom Block 2 (Color 9) is affected.
    # { 'x': 29, 'y': 30 } -> Color 14 appears at r28-31 c27-30. Bottom Block 0 (Color 14) is affected.
    # { 'x': 35, 'y': 30 } -> Color 11 appears at r28-31 c33-36. Bottom Block 3 (Color 11) is affected.
    
    # The logic seems to be:
    # Clicking in the middle area triggers a "spawn" of a color from the bottom area into the middle area.
    # And clicking in the bottom area removes that block.
    
    new_grid = grid.copy()
    if py == 59:
        # Remove block based on X
        if 17 <= px <= 22: # Block 0
            new_grid[56:62, 17:23] = 0
        elif 25 <= px <= 30: # Block 1
            new_grid[56:62, 25:31] = 0
        elif 33 <= px <= 38: # Block 2
            new_grid[56:62, 33:39] = 0
        elif 41 <= px <= 46: # Block 3
            new_grid[56:62, 41:47] = 0
    elif py == 30:
        # Spawn color into middle region
        if 21 <= px <= 24:
            # Color 9 (Block 2) spawns at r28-31 c21-24
            new_grid[28:32, 21:25] = 9
            # Also affect bottom area? "r56c33:4x6 r57c33:4x6..." -> Bottom block becomes 4.
            new_grid[56:62, 33:39] = 4
        elif 27 <= px <= 30:
            # Color 14 (Block 0) spawns at r28-31 c27-30
            new_grid[28:32, 27:31] = 14
            # Bottom block 0 becomes 4.
            new_grid[56:62, 17:23] = 4
        elif 33 <= px <= 36:
            # Color 11 (Block 3) spawns at r28-31 c33-36
            new_grid[28:32, 33:37] = 11
            # Bottom block 3 becomes 4.
            new_grid[56:62, 41:47] = 4
    return new_grid

def is_level_complete(grid):
    # Level complete if the middle region has certain colors or bottom blocks are gone.
    # No win state provided, but usually it's about filling a target area.
    # Let's assume it's completed when all bottom blocks are color 4.
    bottom_blocks = [
        np.sum(grid[57:61, 18:22]) != 0, # Block 0
        np.sum(grid[57:61, 26:30]) != 0, # Block 1
        np.sum(grid[57:61, 34:38]) != 0, # Block 2
        np.sum(grid[57:61, 34:38]) != 0, # Block 3
    ]
    # This is not quite right. If they are all 4 (background), then sum might be non-zero.
    # We check if any of them are NOT background color 4.
    # return np.all(grid[57:61, 18:46] == 4)
    return False # Default to False as no win state was given.