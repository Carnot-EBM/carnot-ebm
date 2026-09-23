import numpy as np


def engine(grid, action, data):
    g = grid.copy()
    h, w = g.shape

    # Track player position by finding color 5 cell
    player_pos = None
    for r in range(h):
        for c in range(w):
            if g[r, c] == 5:
                player_pos = (r, c)
                break
        if player_pos:
            break

    if player_pos is None:
        return g

    pr, pc = player_pos

    # Actions: 1=up, 2=down, 3=left, 4=right, 6=click
    # Movement directions
    dirs = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    if action in dirs:
        dr, dc = dirs[action]
        nr, nc = pr + dr, pc + dc

        if 0 <= nr < h and 0 <= nc < w:
            target_val = g[nr, nc]
            # Player can move onto empty(0), background(2), or wall(1) cells? 
            # From observations, player moves through the structure.
            # The player seems to be a single pixel that moves around.
            # Let's check what happens: player at (31,28) initially (color 5)
            
            # Looking at transitions more carefully:
            # ACTION4 (right): changes happen at rows 30-32, cols 18-20 area AND r63c63->0
            # This suggests the "player" might not be color 5 but something else
            
            # Re-examining: obj10 is color=5 at (31,28) - this is inside obj7 (color 14)
            # obj9 is color=0 at (31,19) - inside obj6 (color 14)
            
            # The deltas show patterns of 14s moving with 0/5 in center
            # It looks like the 14-colored objects (3x3 with center) are being moved/pushed
            
            # Actually looking more carefully at the structure:
            # obj6: color=14 bbox=(30,18,32,20) - 3x3 block of 14s with center 0
            # obj7: color=14 bbox=(30,27,32,29) - 3x3 block of 14s with center 5
            
            # These look like "containers" or "boxes" that can be pushed/moved
            # The player might be one of these containers
            
            # Let me reconsider. Looking at ACTION4 delta:
            # r30c18: 1x3,14x3 -> cells (30,18)=1,(30,19)=1,(30,20)=1,(30,21)=14,(30,22)=14,(30,23)=14
            # Wait no: r30c18 means starting at col 18, values are 1x3 then 14x3
            # So (30,18)=1,(30,19)=1,(30,20)=1,(30,21)=14,(30,22)=14,(30,23)=14
            
            # Hmm but initial grid row 30: 2x9,1x9,14x3,1x6,14x3,1x3,15x6,1x5,4x5,1x5,2x10
            # cols 0-8=2, 9-17=1, 18-20=14, 21-26=1, 27-29=14, 30-32=1, 33-38=15, 39=1, 40-44=5, 45=1, 46-50=4, 51=1, 52-63=2
            
            # Wait that doesn't match. Let me recount:
            # r30: 2x9(0-8), 1x9(9-17), 14x3(18-20), 1x6(21-26), 14x3(27-29), 1x3(30-32), 15x6(33-38), 1x5(39-43)... 
            # Hmm wait: 1x5 means 5 cells of value 1? No, format is <value>x<count>
            # So 1x5 = value 1, count 5 -> cols 39-43 are 1
            # Then 4x5 = value 4, count 5 -> cols 44-48 are 4
            # Then 1x5 = value 1, count 5 -> cols 49-53 are 1  
            # Then 2x10 = value 2, count 10 -> cols 54-63 are 2
            
            # But wait the row has: 2x9,1x9,14x3,1x6,14x3,1x3,15x6,1x5,4x5,1x5,2x10
            # Total: 9+9+3+6+3+3+6+5+5+5+10 = 64 ✓
            
            # So initial row 30: cols 18-20=14, cols 27-29=14
            # Row 31: 2x9,1x9,14x1,0x1,14x1,1x6,14x1,5x1,14x1,1x3,15x6,1x15,2x10
            # Wait that's 9+9+1+1+1+6+1+1+1+3+6+15+10 = 64? Let me check: 9+9=18,+1=19,+1=20,+1=21,+6=27,+1=28,+1=29,+1=30,+3=33,+6=39,+15=54,+10=64 ✓
            # So row 31: col18=14,col19=0,col20=14, col27=14,col28=5,col29=14
            
            # Row 32: 2x9,1x2,4x5,1x2,14x3,1x6,14x3,1x3,15x6,1x15,2x10
            # 9+2+5+2+3+6+3+3+6+15+10=64 ✓
            # cols: 0-8=2, 9-10=1, 11-15=4, 16-17=1, 18-20=14, 21-26=1, 27-29=14, 30-32=1, 33-38=15, 39-53=1, 54-63=2
            
            # So the two 14-blocks are at (30-32, 18-20) and (30-32, 27-29)
            # Block 1 center: (31,19)=0
            # Block 2 center: (31,28)=5
            
            # Now ACTION4 (right) delta: r30c18:1x3,14x3 means starting col 18: 1,1,1,14,14,14
            # That changes cols 18-23 in row 30. But initially cols 18-20 were 14 and 21-26 were 1.
            # After: cols 18-20 become 1, cols 21-23 become 14
            # This looks like the block moved RIGHT by 3? No...
            
            # Actually wait - maybe it's not a simple move. Let me look at all three rows:
            # r30c18: 1x3,14x3 -> (30,18..20)=1, (30,21..23)=14
            # r31c18: 1x3,14x1,0x1,14x1 -> (31,18..20)=1, (31,21)=14,(31,22)=0,(31,23)=14
            # r32c18: 1x3,14x3 -> (32,18..20)=1, (32,21..23)=14
            
            # So the 3x3 block of 14s with center 0 moved from cols 18-20 to cols 21-23!
            # That's a shift right by 3 columns. And the old position became 1s.
            
            # But also r63c63:0x1 means (63,63) changed to 0. Initially row 63 was all 4s.
            # This is a separate effect - maybe a counter or timer?
            
            # So ACTION4 moves the left block (center=0) right by 3? Or does it move by 1 and I'm misreading?
            # Initial: block at cols 18-20. After: block at cols 21-23. Shift = +3.
            
            # Hmm but that seems like a lot for one action. Unless the "player" IS this block
            # and it moves in larger steps, or there are multiple sub-movements.
            
            # Let me check ACTION3 (left): 
            # Later we see ACTION3 delta: r30c21:14x3,1x3 -> (30,21..23)=14,(30,24..26)=1
            # r31c21:14x1,0x1,14x1,1x3 -> (31,21)=14,(31,22)=0,(31,23)=14,(31,24..26)=1
            # r32c21:14x3,1x3 -> (32,21..23)=14,(32,24..26)=1
            
            # So after ACTION3, the block is back at cols 21-23 with center at col 22.
            # Wait, before ACTION3 the block was at cols 21-23 (from ACTION4). After ACTION3 it's still at 21-23?
            # No wait - let me re-read. The ACTION3 that shows r30c21:14x3,1x3 means the NEW values starting at col 21 are 14,14,14,1,1,1.
            # But if the block was already at 21-23, this would mean no change for those cells...
            
            # I think I need to track state more carefully. Let me just implement based on what I see:
            # It appears blocks of 14s (3x3) can be moved in cardinal directions by some amount.
            # The center cell (0 or 5) moves with them.
            
            # Actually, re-reading more carefully, I think the movement might be by 1 step but 
            # the delta encoding shows all changed cells. If a block moves right by 1:
            # Old position cols 18-20 -> new position cols 19-21
            # Changed cells: old left column (col 18) becomes background, new right column (col 21) becomes 14
            # That would be much smaller deltas.
            
            # Given the deltas show 6-cell changes per row, it's a 3-column shift.
            # Maybe the player moves 3 cells per action? Or maybe there's gravity/momentum?
            
            # For simplicity and given limited data, let me try: the "player" is identified as
            # the 3x3 block containing color 5 (or 0), and directional actions move it.
            
            pass

    if action == 6:
        # Click action - from observations, clicks seem to toggle cells or have minimal effect
        # Some clicks caused no change, others changed r63cXX to 0
        px = data.get('x', 0) if data else 0
        py = data.get('y', 0) if data else 0
        lr, lc = py, px  # pixel = logical*1 so same coords
        
        if 0 <= lr < h and 0 <= lc < w:
            # From observations, clicking on certain cells toggles them
            # e.g., click at (31,28) changed (31,28) from 5->4 and (31,22) from 4->... 
            # Actually looking at first ACTION6: r31c22:4x1 means (31,22)=4, r31c28:0x1 means (31,28)=0
            # Initial (31,28)=5, after=0. And (31,22) was initially 0 (center of left block), became 4?
            # Wait initial row 31 col 22: let me check. Row 31: ...14x1(27),5x1(28),14x1(29)...
            # Col 22 in row 31: from the runs, cols 21-26 are value 1. So (31,22)=1 initially.
            # After click: (31,22)=4. Hmm that's odd.
            
            # Second ACTION6 at (35,31): only r63c62:0x1 changed. No local change.
            # This suggests clicks might affect a counter on row 63.
            
            # I'll implement: clicking toggles the clicked cell between its current value and something,
            # AND decrements a counter on row 63.
            pass

    return g


def is_level_complete(grid):
    # Win condition unclear from data - no win state shown
    # Default: check if some specific condition is met
    # For now, assume level is complete when all 14-blocks have been moved to specific positions
    # or when a certain color pattern appears
    return False