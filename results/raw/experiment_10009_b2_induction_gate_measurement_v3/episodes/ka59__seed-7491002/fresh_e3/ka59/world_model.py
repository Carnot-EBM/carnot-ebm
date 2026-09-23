import numpy as np


def engine(grid, action, data):
    g = grid.copy()
    h, w = g.shape

    # Track a counter in bottom-right corner (row 63) that decrements on each action
    # The counter appears to be at row 63, starting from col 63 going left
    # Looking at the pattern: r63c63:0x1 means cell (63,63) changed to 0
    # Then r63c62:0x1, r63c61:0x1, etc. - it's filling cells with 0 from right to left

    # Actually looking more carefully:
    # Initial: r63 is all 4s
    # After actions, individual cells in row 63 change to 0
    # This seems like a step counter or move tracker

    # Let me re-examine the structure:
    # The main game area has rooms/structures made of colors 1, 2, 4, 5, 14, 15
    # Actions seem to manipulate objects within these structures

    # Key observations from transitions:
    # ACTION4: moves something related to color 14 objects (the "doors" or "gates")
    # ACTION3: also manipulates color 14 objects  
    # ACTION1/ACTION2: move things vertically
    # ACTION6: click action that toggles small single-pixel objects (color 0 and 5)

    # Looking at the pattern more carefully:
    # obj6 (color=14) at bbox=(30,18,32,20) - a 3x3 block with center pixel being 0 (obj9)
    # obj7 (color=14) at bbox=(30,27,32,29) - a 3x3 block with center pixel being 5 (obj10)
    
    # These look like "slots" or "sockets" where items can be placed
    
    # The 3x3 blocks of color 14 have a center cell that is either 0 (empty) or has an item
    # ACTION6 clicks on these centers to toggle/place items

    # Let me trace through the actions:
    # Initial state: 
    #   obj6 at (30-32, 18-20): 14s around, center (31,19)=0
    #   obj7 at (30-32, 27-29): 14s around, center (31,28)=5

    # ACTION4 (level 0->0): r30c18:1x3,14x3 r31c18:1x3,14x1,0x1,14x1 r32c18:1x3,14x3 r63c63:0x1
    # This changes the area around (30-32, 18-20):
    # Before: 14,14,14 / 14,0,14 / 14,14,14 (the 14-block with 0 center)
    # After:  1,1,1 / 1,14,0,14... wait let me re-read
    
    # Actually r30c18 means row 30, starting col 18. The values are 1x3,14x3 = cols 18,19,20 become 1, then cols 21,22,23 become 14? No that doesn't make sense for a 3-wide block.
    
    # Wait - the delta format is: r<row>c<col0>:<v0>x<n0>,<v1>x<n1>
    # So r30c18:1x3,14x3 means at row 30, starting at col 18: 3 cells of value 1, then 3 cells of value 14
    # That's cols 18-20 = 1, cols 21-23 = 14
    
    # But initially row 30 was: 2x9,1x9,14x3,1x6,14x3,1x3,15x6,1x5,4x5,1x5,2x10
    # Let me count: 9+9+3+6+3+3+6+5+5+5+10 = 64 ✓
    # So cols 0-8=2, 9-17=1, 18-20=14, 21-26=1, 27-29=14, 30-32=1, 33-38=15, 39-43=1, 44-48=4, 49-53=1, 54-63=2
    
    # After ACTION4: r30c18:1x3,14x3 means cols 18-20 become 1, cols 21-23 become 14
    # So the 14-block shifted RIGHT by 3? From cols 18-20 to cols 21-23?
    
    # And r31c18:1x3,14x1,0x1,14x1 means cols 18-20=1, col21=14, col22=0, col23=14
    # Initially row 31: 2x9,1x9,14x1,0x1,14x1,1x6,14x1,5x1,14x1,1x3,15x6,1x15,2x10
    # Cols: 0-8=2, 9-17=1, 18=14, 19=0, 20=14, 21-26=1, 27=14, 28=5, 29=14, 30-32=1, 33-38=15, 39-53=1, 54-63=2
    
    # After: cols 18-20=1, 21=14, 22=0, 23=14
    # So the pattern 14,0,14 moved from cols 18-20 to cols 21-23!
    
    # r32c18:1x3,14x3 means cols 18-20 become 1, cols 21-23 become 14
    # Initially row 32: 2x9,1x2,4x5,1x2,14x3,1x6,14x3,1x3,15x6,1x15,2x10
    # Cols: 0-8=2, 9-10=1, 11-15=4, 16-17=1, 18-20=14, 21-26=1, 27-29=14, 30-32=1, 33-38=15, 39-53=1, 54-63=2
    
    # After: cols 18-20=1, 21-23=14. So the 14-block at cols 18-20 moved to 21-23.

    # So ACTION4 moves the left 14-block (obj6) RIGHT by 3 columns!
    
    # Now let's check ACTION3 which seems to move it back:
    # ACTION3: r30c21:14x3,1x3 r31c21:14x1,0x1,14x1,1x3 r32c21:14x3,1x3
    # This means at row 30 col 21: 3 cells of 14, then 3 cells of 1 → cols 21-23=14, cols 24-26=1
    # At row 31 col 21: 14,0,14,1 → cols 21=14, 22=0, 23=14, 24=1
    # At row 32 col 21: 14,14,14,1 → cols 21-23=14, col 24=1
    
    # Wait but after ACTION4, the state was:
    # Row 30: ...cols 18-20=1, 21-23=14...
    # After ACTION3: cols 21-23=14 (unchanged), cols 24-26=1 (was already 1)
    
    # Hmm that doesn't show a move. Let me re-read.
    
    # Actually wait - I need to track state across multiple actions. Let me be more careful.

    # Let me reconsider. The key objects are:
    # - Two "socket" structures made of color 14 in a ring pattern (3x3 with hollow center)
    # - Small items (color 0 or 5) that can be placed in the centers
    # - Color 4 blocks (3x3 squares) 
    # - Color 1 walls forming rooms
    # - Color 2 background/border
    # - Color 15 vertical bars
    # - Color 5 small pixels inside color 4 blocks

    # The game seems to involve moving the 14-rings and placing items in them.

    # Given the complexity and limited observations, let me implement based on patterns:

    # ACTION1 = Up, ACTION2 = Down, ACTION3 = Left, ACTION4 = Right
    # ACTION6 = Click (toggle/place item at clicked position)

    # The 14-ring objects seem to be movable entities.
    # Looking at the deltas, they shift by 3 columns when moved left/right.

    # For the counter in row 63: it decrements from right to left (col 63, 62, 61...)
    # Each action that changes something also sets one more cell in row 63 to 0.

    # Let me implement a simpler model:

    # Find all "movable" 14-ring structures and track their positions
    # The rings are 3x3 patterns of 14s with a non-14 center

    def find_14_rings(g):
        """Find 3x3 ring patterns of color 14."""
        rings = []
        for r in range(h - 2):
            for c in range(w - 2):
                if g[r, c] == 14 and g[r+1, c] == 14 and g[r+2, c] == 14 and \
                   g[r, c+2] == 14 and g[r+1, c+2] == 14 and g[r+2, c+2] == 14 and \
                   g[r, c+1] == 14 and g[r+2, c+1] == 14 and \
                   g[r+1, c+1] != 14:
                    rings.append((r, c))
        return rings

    def move_ring(g, ring_pos, dr, dc):
        """Move a 14-ring by (dr, dc)."""
        r0, c0 = ring_pos
        r1, c1 = r0 + dr, c0 + dc
        
        # Check bounds
        if r1 < 0 or r1 > h - 3 or c1 < 0 or c1 > w - 3:
            return False
        
        # Check destination is clear (all cells should be the "background" color for that area)
        # The background inside rooms seems to be color 1
        dest_ok = True
        for dr_ in range(3):
            for dc_ in range(3):
                val = g[r1 + dr_, c1 + dc_]
                if val == 14:
                    dest_ok = False
                    break
            if not dest_ok:
                break
        
        if not dest_ok:
            return False
        
        # Save center value
        center_val = g[r0 + 1, c0 + 1]
        
        # Clear old position (set to surrounding background)
        # Determine what the background color is at the old position
        # Look at adjacent cells outside the ring
        bg_color = 1  # default room interior color
        
        for dr_ in range(3):
            for dc_ in range(3):
                g[r0 + dr_, c0 + dc_] = bg_color
        
        # Place ring at new position
        for dr_ in range(3):
            for dc_ in range(3):
                if dr_ == 1 and dc_ == 1:
                    g[r1 + dr_, c1 + dc_] = center_val
                else:
                    g[r1 + dr_, c1 + dc_] = 14
        
        return True

    def update_counter(g, changed):
        """Update the step counter in row 63."""
        if not changed:
            return
        # Find the rightmost non-zero cell in row 63 that should become 0
        # The pattern shows cells changing from 4 to 0, going right to left
        # Actually looking again: initial row 63 is all 4s
        # After first action: r63c63:0x1 → cell (63,63) becomes 0
        # After second: r63c62:0x1 → cell (63,62) becomes 0
        # So it's filling from right to left with 0s
        
        # Find the next cell to set to 0 (rightmost 4 in row 63)
        row63 = g[63]
        for c in range(w - 1, -1, -1):
            if row63[c] == 4:
                row63[c] = 0
                break

    # Determine what changed based on action
    changed = False

    if action in [1, 2, 3, 4]:
        # Directional movement of 14-rings
        rings = find_14_rings(g)
        
        if action == 1:  # Up
            dr, dc = -1, 0
        elif action == 2:  # Down
            dr, dc = 1, 0
        elif action == 3:  # Left
            dr, dc = 0, -1
        elif action == 4:  # Right
            dr, dc = 0, 1

        moved_any = False
        for ring_pos in rings:
            if move_ring(g, ring_pos, dr * 3, dc * 3):
                moved_any = True
        
        if moved_any:
            changed = True
            update_counter(g, True)

    elif action == 6 and data is not None:
        # Click action - toggle item at clicked position
        px, py = data['x'], data['y']
        # Pixel coords = logical coords (pixel = logical*1)
        r, c = py, px
        
        if 0 <= r < h and 0 <= c < w:
            # Check if this is a center of a 14-ring
            # If so, toggle between 0 and 5
            is_center = False
            for rr in range(max(0, r-1), min(h, r+2)):
                for cc in range(max(0, c-1), min(w, c+2)):
                    pass
            
            # Simpler: check if the cell is surrounded by 14s (it's a ring center)
            if r > 0 and r < h-1 and c > 0 and c < w-1:
                surrounding = [g[r-1,c], g[r+1,c], g[r,c-1], g[r,c+1]]
                if all(s == 14 for s in surrounding):
                    # Toggle: 0 -> 5 or 5 -> 0
                    if g[r, c] == 0:
                        g[r, c] = 5
                        changed = True
                    elif g[r, c] == 5:
                        g[r, c] = 0
                        changed = True

    return g


def is_level_complete(grid):
    """Check if level is complete."""
    # Based on observations, no win state was shown.
    # A reasonable heuristic: all items are placed correctly in their sockets.
    # Without clear win condition data, return False unless specific pattern matches.
    
    # Check if there's a specific completion pattern
    # For now, assume completion when certain conditions are met
    # Since we don't have explicit win state data, use a conservative check
    
    h, w = grid.shape
    
    # One possible win condition: all 14-rings have color 5 centers (items collected)
    # Or: the counter row is fully filled with 0s
    
    # Conservative: return False by default since no win state observed
    return False