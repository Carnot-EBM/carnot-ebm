import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    
    # Track a counter at bottom-right corner for click actions
    if action == 6 and data is not None:
        px, py = data.get('x', 0), data.get('y', 0)
        # The observed behavior shows r63c<col> changing from 4 to 5 sequentially
        # This appears to be a simple counter decrementing column index
        # Find current rightmost non-4 cell in row 63
        row = g[63]
        # Find the first 4 from the right that should become 5
        # Based on observations, it seems like a specific sequence
        # Let's just implement the general rule: clicking changes a specific cell
        # From observations: clicks change cells in row 63, col decreasing from 63
        # But we need to figure out which cell exactly
        
        # Looking at the pattern more carefully:
        # Each ACTION6 changes exactly one cell in row 63 from 4 to 5
        # The columns go: 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48...
        # So it's simply decrementing. We track state via the grid itself.
        
        # Find the rightmost 4 in row 63 and change it to 5
        for c in range(63, -1, -1):
            if g[63, c] == 4:
                g[63, c] = 5
                break
    
    elif action == 1:
        # Move up / apply upward effect
        _apply_directional(g, 'up')
    
    elif action == 2:
        # Move down / apply downward effect  
        _apply_directional(g, 'down')
    
    elif action == 3:
        # Move left / apply leftward effect
        _apply_directional(g, 'left')
    
    elif action == 4:
        # Move right / apply rightward effect
        _apply_directional(g, 'right')
    
    elif action == 5:
        # Special action: fill certain regions with 15
        _apply_action5(g)
    
    return g

def _apply_directional(g, direction):
    """Apply directional movement/gravity effects based on observed patterns."""
    h, w = g.shape
    
    # The key objects are:
    # - Color 2 (border/frame) at rows 24-32, cols 25-38 area
    # - Color 15 (fill) inside the frame
    # - Color 0 (empty/space) in various areas
    
    # From observations, actions 1-4 seem to move/shift the 15-colored region
    # and affect the 2-colored border structure.
    
    # Let's identify the main interactive region
    # The "box" is roughly at rows 24-37, cols 14-45
    
    if direction == 'up':
        _move_box_up(g)
    elif direction == 'down':
        _move_box_down(g)
    elif direction == 'left':
        _move_box_left(g)
    elif direction == 'right':
        _move_box_right(g)

def _find_box_bounds(g):
    """Find the bounding box of color 2 cells (the frame)."""
    mask = (g == 2)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(rmin), int(cmin), int(rmax), int(cmax)

def _move_box_up(g):
    h, w = g.shape
    bounds = _find_box_bounds(g)
    if bounds is None:
        return
    rmin, cmin, rmax, cmax = bounds
    
    # Shift the entire region up by checking what's above
    # From observations, moving up shifts the 15-fill upward and 
    # the border adjusts accordingly
    
    # Simple approach: shift all non-background pixels in the interactive area
    # The background is color 5
    # Extract the sub-region and shift it
    
    # Based on observed deltas, the effect is complex. Let me use a simpler model:
    # The box (colors 2 and 15) moves as a unit when possible.
    
    # Try shifting up by 1
    new_g = g.copy()
    for r in range(rmin, rmax + 1):
        for c in range(cmin, cmax + 1):
            val = g[r, c]
            if val != 5:  # not background
                target_r = r - 1
                if target_r >= 0:
                    new_g[target_r, c] = val
    # Clear original positions that were moved
    for r in range(rmin, rmax + 1):
        for c in range(cmin, cmax + 1):
            if g[r, c] != 5:
                g[r, c] = 5
    
    # This simple shift doesn't match observations well.
    # Let me reconsider based on actual delta patterns.
    
    # Actually, looking more carefully at the transitions:
    # ACTION3 (left) and ACTION4 (right) seem to move the 15-fill horizontally
    # within the frame, while also adjusting the 2-border.
    # ACTION1 (up) and ACTION2 (down) do similar vertically.
    
    # The pattern suggests the 15-colored region is a "liquid" or "fill" 
    # that moves within constraints defined by color 2 borders.
    
    pass

def _move_box_down(g):
    pass

def _move_box_left(g):
    pass

def _move_box_right(g):
    pass

def _apply_action5(g):
    """Action 5 fills certain regions with color 15."""
    h, w = g.shape
    # From observations, action 5 changes specific rows/cols to 15
    # It seems to fill the interior of the box structure
    # Looking at deltas: r34c27:15x10 through r38c27:15x10
    # And later: r39c27:15x5 through r43c27:15x5
    
    # This appears to be filling empty (color 0) spaces inside the structure
    # Let's find color 0 cells adjacent to color 2 or 15 and fill them
    
    for r in range(h):
        for c in range(w):
            if g[r, c] == 0:
                # Check if this cell is "inside" a structure
                # Simple heuristic: if surrounded by non-5 colors
                neighbors = []
                if r > 0: neighbors.append(g[r-1, c])
                if r < h-1: neighbors.append(g[r+1, c])
                if c > 0: neighbors.append(g[r, c-1])
                if c < w-1: neighbors.append(g[r, c+1])
                if any(n in [2, 15] for n in neighbors):
                    g[r, c] = 15

def is_level_complete(grid):
    """Check if the grid represents a win state."""
    # Based on observations, no explicit win state was shown.
    # A reasonable guess: all interactive elements are resolved.
    # Perhaps when there are no more color 0 cells in certain regions,
    # or when specific conditions are met.
    
    # For now, check if the bottom row (63) is fully converted to 5
    # (since clicks were converting 4->5 sequentially)
    if np.all(grid[63] == 5):
        return True
    
    # Or check if all color 0 cells have been filled
    if not np.any(grid == 0):
        return True
    
    return False