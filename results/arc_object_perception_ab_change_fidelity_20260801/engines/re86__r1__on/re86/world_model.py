import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION4: Move right? (The delta shows a shift in columns for color 9 and others)
    # ACTION5: Toggle/Interact? (Changes specific cells)
    # ACTION1: Move up? (The delta shows a shift in rows for row indices decreasing)
    # Let's refine movement rules.
    
    # Based on the observations, it's a puzzle game where an "agent" or "player" object (color 11)
    # is moving through a maze of walls (color 5) and interacting with blocks own colors.
    # The same-shape objects are tracked.
    # obj9 (color 11) is likely the player.
    # Color 9 represents movable blocks.
    #<|channel>thought
    # ACTION 4: Right
    # ACTION 1: Up
    # ACTION 2: Left
    # ACTION 3: Down
    # ACTION 6: Click
    
    # Looking at the deltas:
    # ACTION 4 shifts things to the right by 3 pixels.
    # ACTION 1 shifts things up by some amount.
    # In ACTION 1, we see r13c21 -> r10c21 etc., which is a vertical move.
    
    # Find the agent (color 11). It's usually a block/cluster.
    # We need to identify the 'active' part of the level.
    # The grid contains color 5 as background/walls.
    # Let's find all coordinates of color 11.
    
    agent_coords = np.argwhere(grid == 11)
    if len(agent_coords) == 0:
        return grid.copy()

    # The "player" seems to be the larger connected component of color 11.
    # But in this game, it might just be any cell of color 11 moving.
    # Actually, looking at the INITIAL GRID and OBJECT STRUCTURE, obj9 is the main player.
    # Color 11 cells are also present inside color 4 blocks.
    
    # To simplify, let's assume movement applies to the largest cluster of color 11.
    # However, the deltas show that when the player moves, other things move too.
    # Specifically, if the player pushes a block of color 9, that block moves.
    
    # Movement vectors:
    # ACTION 1: Up (dy=-3 or similar)
    # ACTION 2: Left (dx=-3)
    # ACTION 3: Down (dy=3)
    # ACTION 4: Right (dx=3)
    
    # Let's check ACTION 4 again: r11c39 -> r11c42. dx = +3.
    # ACTION 1: r13c21 -> r10c21. dy = -3.
    
    move_map = {
        1: (-3, 0), # Up
        2: (0, -3), # Left
        3: (3, 0),  # Down
        4: (0, 3),  # Right
    }
    
    if action not in move_map:
        return grid.copy()
    
    dy, dx = move_map[action]
    
    # Identify all "movable" objects. These are connected components of colors other than 5 and 15.
    # But the deltas show only color 11 (player) and color 9 (blocks) moving.
    # Color 4 blocks also seem to be static walls or containers.
    
    # To implement this simply:
    # 1. Find all cells that are NOT background (color 5).
    # 2. For each cell, if it's part of a movable object, try to shift it.
    # 3. Check for collisions with walls (color 5).
    
    # In these specific transitions, the player is at x=21, y=~27.
    # When ACTION 4 (Right) happens, the block at c=39 moves to c=42.
    # This means the player pushes the block.
    
    # Let's find the player cluster (largest component of color 11).
    # Then check if there's a block of color 9 in front of them.
    
    new_grid = grid.copy()
    
    # Simplified logic based on observed deltas:
    # The movement is discrete steps of 3 units.
    # Only certain objects move.
    
    # Instead of complex physics, let's just apply the translation to any non-wall cell
    # that is "pushed" by the agent.
    
    # But wait, the deltas show the player moving and blocks moving simultaneously.
    # Let's use a simpler approach: identify all cells that are not color 5 or 15.
    # If they can move without hitting a wall (color 5), move them? No, only if pushed.
    
    # Actually, looking at the laout, it looks like a Sokoban variant.
    # Player (11) pushes Blocks (9).
    
    # Find player center
    player_cells = np.argwhere(grid == 11)
    if len(player_cells) == 0: return grid
    py, px = np.mean(player_cells, axis=0).astype(int)
    
    # Check for block in direction of movement
    block_found = False
    check_y, check_x = py + dy, px + dx
    if 0 <= check_y < 64 and 0 <= check_x < 64:
        if grid[check_y, check_x] == 9:
            block_found = True
            
    # Move everything that is part of the "movable" set
    # For this specific level, we just need to shift the agent and any blocks he's pushing.
    
    # Let's find all cells of color 11 and 9 and try to shift them.
    # If they are connected to the player or pushed by something moving, they move.
    
    # To be very simple and match the deltas:
    # Shift all color 11 and color 9 cells by (dy, dx) if they don't hit a wall.
    # But only those that are actually moving in the delta.
    
    # The observed ACTION 4 moves the block at c=39 to c=42.
    # The player is at x=21. The distance is 18 units.
    # This suggests a "remote push" or a long-range interaction.
    
    # Looking closer at INITIAL GRID:
    # Player (obj9) is bbox=(16, 10, 38, 32). Center ~ (27, 21).
    # Block (obj7) is bbox=(11, 39, 23, 39). Center ~ (17, 39).
    # When ACTION 4 happens, obj7 moves from x=39 to x=42.
    # This means when the player moves Right, any block of color 9 in the same row(s) to the right moves too?
    
    # Let's try this rule:
    # If action is MoveRight, all blocks of color 9 with x > player_x move Right by 3.
    # If action is MoveLeft, all blocks of color 9 with x < player_x move Left by 3.
    # Similarly for Up/Down.
    
    movable_cells = np.argwhere((grid == 11) | (grid == 9))
    for r, c in movable_cells:
        # Determine if this cell should move based on its position relative to player center
        should_move = False
        if action == 4 and c > px: should_move = True # Right
        elif action == 2 and c < px: should_move = True # Left
        elif action == 3 and r > py: should_move = True # Down
        elif action == 1 and r < py: should_move = True # Up
        elif (grid[r, c] == 11): should_move = True # Player always moves
        
        if should_move:
            nr, nc = r + dy, c + dx
            if 0 <= nr < 64 and 0 <= nc < 64:
                if grid[nr, nc] != 5: # Not a wall
                    new_grid[nr, nc] = grid[r, c]
                    new_grid[r, c] = 5 # Leave background
                    
    return new_grid

def is_level_complete(grid):
    # Win state usually involves moving blocks to specific locations or collecting them.
    # In the observed data, we don't have a win state, but typically it's when color 1 is at some pos.
    # Or maybe when all color 9 are gone?
    # Let's check if any cell of color 1 exists in a certain area.
    # Based on INITIAL GRID, color 1 is at (63, 59-63).
    # The deltas show ACTION 1 reducing the count of color 1 cells from 5 down to 1.
    # This suggests that "collecting" something reduces the size of the goal/exit.
    # Level complete might be when no more movable blocks exist or player reaches exit.
    return False # No win state provided in observations.

import numpy as np

def is_level_complete(grid):
    # The same color must be the same color
    # The win state is when the same color is a rectangle
    # A simple general rule: a block of same-color pixels own a
    thought//no_think
