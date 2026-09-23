import numpy as np


def _find_player(grid):
    """Find the player position (color 5)."""
    positions = np.argwhere(grid == 5)
    if len(positions) > 0:
        return tuple(positions[0])
    return None


def _get_cell(grid, r, c):
    h, w = grid.shape
    if 0 <= r < h and 0 <= c < w:
        return int(grid[r, c])
    return -1


def _set_cell(grid, r, c, val):
    h, w = grid.shape
    if 0 <= r < h and 0 <= c < w:
        grid[r, c] = val


def engine(grid, action, data):
    g = grid.copy()
    h, w = g.shape

    # Track a counter on row 63 for click actions
    # The pattern shows that each ACTION6 decrements a counter starting from col 63 going left
    # We'll simulate by finding the rightmost non-zero cell in row 63 and setting it to 0

    if action == 6:
        # Click action: find the rightmost non-zero pixel in row 63 and set to 0
        row63 = g[63].copy()
        for c in range(w - 1, -1, -1):
            if row63[c] != 0:
                g[63, c] = 0
                break
        return g

    # Find player position (color 5)
    player_pos = _find_player(g)
    if player_pos is None:
        return g

    pr, pc = player_pos

    # Determine movement direction based on action
    # Action 1=up, 2=down, 3=left, 4=right
    dr_dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    if action not in dr_dc:
        return g

    dr, dc = dr_dc[action]
    nr, nc = pr + dr, pc + dc

    target_val = _get_cell(g, nr, nc)

    # If target is out of bounds or a wall (color 1), don't move
    if target_val == 1:
        return g

    # The player moves into the target cell. The player's old cell becomes color 1 (wall).
    # The target cell's content is replaced by the player (color 5).
    # But we also need to handle the "frame" objects (color 14 with center pixel).

    # Looking at the transitions more carefully:
    # When moving right (action 4): the 3x3 frame at the destination gets its left column changed to 1
    # and the player appears in the center.
    # When moving left (action 3): the 3x3 frame at the destination gets its right column changed to 1
    # and the player appears in the center.

    # Let me re-analyze:
    # Initial state has two 3x3 frames made of color 14 with a center pixel (0 for left, 5 for right).
    # obj6: color=14 bbox=(30,18,32,20) - left frame, center at (31,19) is color 0
    # obj7: color=14 bbox=(30,27,32,29) - right frame, center at (31,28) is color 5 (player!)

    # So the player starts at (31,28), inside the right frame.
    # The left frame has center (31,19) which is color 0 (empty/dark).

    # ACTION4 (move right from 31,28): 
    #   r30c18: 1x3,14x3 -> cols 18-20 become [1,1,1,14,14,14]... wait that's 6 cells
    #   Actually r30c18 means starting at col 18, and the values are 1x3 then 14x3 = 6 cells total
    #   That covers cols 18-23? No, 3+3=6 cells: cols 18,19,20 get value 1, cols 21,22,23 get value 14
    #   
    # Wait, let me re-read the delta format. "r30c18:1x3,14x3" means starting at row 30, col 18,
    # the changed cells have new values: first 3 cells are 1, next 3 cells are 14.
    # So cols 18,19,20 -> 1 and cols 21,22,23 -> 14.

    # Hmm, but the left frame was at cols 18-20 (bbox x0=18,x1=20). After ACTION4, 
    # cols 18-20 become 1 (wall) and cols 21-23 become 14.
    # This looks like the frame is being PUSHED to the right!

    # Let me reconsider: maybe the player pushes frames.
    # Player at (31,28), moves right. The right frame (cols 27-29) gets pushed?
    # But the delta shows changes at cols 18-23, not 27-29.

    # Actually wait - looking again at the initial grid:
    # r30: ...14x3... at position after "1x6" which starts at col 9+15+9=33... 
    # Let me recount row 30: 2x9,1x9,14x3,1x6,14x3,1x3,15x6,1x5,4x5,1x5,2x10
    # cols: 0-8(2), 9-17(1), 18-20(14), 21-26(1), 27-29(14), 30-32(1), 33-38(15), 39-43(1), 44-48(4), 49-53(1), 54-63(2)
    
    # So left frame is at cols 18-20, right frame at cols 27-29. Player (color 5) at (31,28).
    
    # ACTION4 delta: r30c18:1x3,14x3 -> cols 18-20 become 1, cols 21-23 become 14
    # This means the LEFT frame moved from cols 18-20 to cols 21-23!
    # And the player... where did it go? The delta doesn't show color 5 appearing anywhere new.
    # 
    # Wait, let me check r31: "r31c18:1x3,14x1,0x1,14x1" = cols 18-20->1, col21->14, col22->0, col23->14
    # That's only 6 cells changed in row 31 starting at col 18.
    # Original row 31 cols 18-23: 14,0,14,1,1,1 (from initial: ...14x1,0x1,14x1,1x6...)
    # After: 1,1,1,14,0,14
    
    # So the left frame (which was 14,0,14 pattern) moved right by 3 columns!
    # Cols 18-20 were 14,0,14 -> now 1,1,1
    # Cols 21-23 were 1,1,1 -> now 14,0,14

    # This is a PUSH! The player pushed the left frame to the right.
    # But wait, the player is at (31,28), which is in the RIGHT frame area. How does moving right push the LEFT frame?

    # I think I'm overcomplicating this. Let me look at it differently.
    # Maybe action 4 doesn't move the player but instead moves/pushes objects.
    # Or maybe the "player" isn't color 5 - maybe color 5 is just a marker.

    # Let me reconsider the whole game mechanics based on all transitions:

    # Looking at ACTION1 (up): changes at r27c15-r32c15 area
    # The 3x3 block of color 4 at rows 27-29, cols 15-17 gets changed... 
    # Actually looking more carefully at the object structure:
    # obj11: color=4 bbox=(32,11,36,15) - a 5x5 block of color 4
    # obj12: color=1 bbox=(33,12,35,14) - inside that, a 3x3 of color 1
    # So there's a 5x5 color-4 box with a 3x3 color-1 interior at rows 32-36, cols 11-15

    # And obj4: color=4 bbox=(26,44,30,48) - another 5x5 color-4 box at rows 26-30, cols 44-48
    # obj5: color=1 bbox=(27,45,29,47) - 3x3 color-1 interior

    # These look like "doors" or "portals" - 5x5 boxes with hollow centers.

    # I think the game involves:
    # - A player (color 5) that can move in 4 directions
    # - Frames/objects (color 14) that can be pushed
    # - The counter on row 63 tracks something (maybe moves made)

    # Given the complexity and limited observations, let me implement what I can see:

    # Key observations from transitions:
    # 1. ACTION6 (click): sets one pixel in row 63 to 0 (rightmost non-zero going left)
    # 2. Directional actions seem to push/move the 3x3 frames (color 14)
    # 3. The player (color 5) position changes based on movement

    # Let me try a simpler model:
    # - Player is color 5
    # - Moving into a frame pushes it
    # - Moving into empty space just moves the player
    # - Row 63 has a countdown timer

    # Actually, re-examining: after ACTION4, where is the player?
    # The delta for ACTION4 doesn't show any new color-5 cell being created.
    # This means either the player didn't move, or the player's old position was already 
    # accounted for in the changed cells.

    # Initial player at (31,28). After ACTION4, if player moved to (31,29), then:
    # (31,28) would need to change from 5 to something else, and (31,29) from 1 to 5.
    # But the delta only shows changes at cols 18-23 in rows 30-32. No changes at col 28!
    # So the PLAYER DID NOT MOVE with ACTION4. Instead, the left frame was pushed right by 3.

    # This suggests that directional actions don't move the player directly but instead
    # push objects in that direction. Or maybe the player IS one of the frames.

    # New theory: The "player" is actually the RIGHT frame (color 14 with center 5).
    # When you press a direction, the frame moves/pushes other things.
    # The center pixel (5) is just the "active" indicator within the frame.

    # Let me look at ACTION3 (left):
    # r30c21:14x3,1x3 -> cols 21-23 become 14, cols 24-26 become 1
    # r31c21:14x1,0x1,14x1,1x3 -> cols 21->14, 22->0, 23->14, 24-26->1
    # r32c21:14x3,1x3 -> cols 21-23 become 14, cols 24-26 become 1
    
    # Before ACTION3, after previous actions, the left frame was at cols 21-23 (pushed there by ACTION4).
    # After ACTION3, it's back at cols 21-23? No wait...
    
    # Let me trace more carefully. This is getting very complex without being able to 
    # actually run the code. Let me implement a best-effort model based on the patterns I see.

    # SIMPLIFIED MODEL:
    # - The game has a "cursor" or active element that responds to directional inputs
    # - Directional inputs push/move 3x3 frames in the direction pressed
    # - Clicks decrement a counter on row 63
    # - Win condition: all counters reach 0 or some specific arrangement

    # Given the extreme complexity and limited data, I'll implement the most observable rules:

    if action == 6:
        # Already handled above
        return g

    # For directional actions, find the relevant frame and push it
    # The frames are 3x3 blocks of color 14 with a center pixel

    # Find all 3x3 frames of color 14
    def find_frames(g):
        frames = []
        for r in range(h - 2):
            for c in range(w - 2):
                # Check if this is top-left corner of a 3x3 frame
                if (g[r, c] == 14 and g[r, c+1] == 14 and g[r, c+2] == 14 and
                    g[r+1, c] == 14 and g[r+1, c+2] == 14 and
                    g[r+2, c] == 14 and g[r+2, c+1] == 14 and g[r+2, c+2] == 14):
                    center_val = int(g[r+1, c+1])
                    frames.append((r, c, center_val))
        return frames

    frames = find_frames(g)

    if not frames:
        return g

    # Determine which frame to move based on action direction
    # Action 1=up, 2=down, 3=left, 4=right
    # The frame that gets moved seems to be the one closest to the player or 
    # the one in the path of movement

    # From observations, it seems like ALL frames get pushed in the given direction
    # Let me check: ACTION4 pushes left frame right by 3 cols. Does it also push right frame?
    # The delta for ACTION4 only shows changes at cols 18-23 (left frame area).
    # So only ONE frame moves per action.

    # Which frame? It seems to be the LEFTMOST frame when moving right, 
    # or some specific selection logic.

    # Actually, I think the rule might be simpler than I thought:
    # The directional key moves the NEAREST frame in that direction.
    # Or perhaps there's a "selected" frame concept.

    # Given the difficulty, let me just implement the click counter and basic frame pushing:

    # For now, implement: directional actions push the nearest frame in that direction by 3 cells
    if action in [1, 2, 3, 4]:
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]

        # Find the frame closest to center of grid that can move in this direction
        # From observations, it appears the leftmost/topmost frame is selected
        
        # Select the first frame (top-left most) that can be pushed
        target_frame = None
        for fr, fc, cv in frames:
            nr_fr = fr + dr * 3
            nc_fc = fc + dc * 3
            # Check if destination is valid (within bounds and not overlapping walls)
            if 0 <= nr_fr < h - 2 and 0 <= nc_fc < w - 2:
                target_frame = (fr, fc, cv)
                break

        if target_frame is None:
            return g

        fr, fc, cv = target_frame
        new_r = fr + dr * 3
        new_c = fc + dc * 3

        # Extract the 3x3 frame pattern
        old_pattern = g[fr:fr+3, fc:fc+3].copy()

        # Clear old position (set to color 1 which is the background wall color inside the room)
        g[fr:fr+3, fc:fc+3] = 1

        # Place at new position
        g[new_r:new_r+3, new_c:new_c+3] = old_pattern

    return g


def is_level_complete(grid):
    """Check if level is complete. Based on observations, the win state likely 
    involves all row-63 pixels being 0 or a specific arrangement."""
    # From the transitions, row 63 starts as all 4s and gets decremented by clicks.
    # Win might be when row 63 is all zeros, or when