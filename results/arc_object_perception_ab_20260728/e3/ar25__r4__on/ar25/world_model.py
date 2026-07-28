import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 3:
        # Action 3: Toggle specific rows (15-23) at specific columns
        # Based on observed deltas, this action toggles rows 15-23 at columns 6, 15, 45, 54
        # The pattern is: toggle 3x3 blocks or similar structures
        # From deltas: r15c6, r15c15, r15c45, r15c54 are toggled
        # The toggling seems to affect rows 15-23
        
        # Identify the toggle columns based on the pattern
        toggle_cols = [6, 15, 45, 54]
        toggle_rows = range(15, 24)
        
        for r in toggle_rows:
            for c in toggle_cols:
                # Toggle the cell
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 9
                else:
                    new_grid[r, c] = 5
                # Also toggle adjacent cells in a 3x3 pattern
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if new_grid[nr, nc] == 5:
                                new_grid[nr, nc] = 9
                            else:
                                new_grid[nr, nc] = 5
                                
    elif action == 2:
        # Action 2: Toggle specific rows (18-23) at specific columns
        # Based on observed deltas, this action toggles rows 18-23 at columns 3, 51, 54
        # The pattern is: toggle 3x3 blocks or similar structures
        # From deltas: r18c3, r18c51, r18c54 are toggled
        
        # Identify the toggle columns based on the pattern
        toggle_cols = [3, 51, 54]
        toggle_rows = range(18, 24)
        
        for r in toggle_rows:
            for c in toggle_cols:
                # Toggle the cell
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 9
                else:
                    new_grid[r, c] = 5
                # Also toggle adjacent cells in a 3x3 pattern
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if new_grid[nr, nc] == 5:
                                new_grid[nr, nc] = 9
                            else:
                                new_grid[nr, nc] = 5
                                
    elif action == 1:
        # Action 1: Toggle specific rows (15-23) at specific columns
        # Based on observed deltas, this action toggles rows 15-23 at columns 3, 12, 48, 57
        # The pattern is: toggle 3x3 blocks or similar structures
        # From deltas: r15c3, r15c12, r15c48, r15c57 are toggled
        
        # Identify the toggle columns based on the pattern
        toggle_cols = [3, 12, 48, 57]
        toggle_rows = range(15, 24)
        
        for r in toggle_rows:
            for c in toggle_cols:
                # Toggle the cell
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 9
                else:
                    new_grid[r, c] = 5
                # Also toggle adjacent cells in a 3x3 pattern
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if new_grid[nr, nc] == 5:
                                new_grid[nr, nc] = 9
                            else:
                                new_grid[nr, nc] = 5
                                
    elif action == 4:
        # Action 4: Toggle specific rows (15-23) at specific columns
        # Based on observed deltas, this action toggles rows 15-23 at columns 9, 18, 51, 60
        # The pattern is: toggle 3x3 blocks or similar structures
        # From deltas: r15c9, r15c18, r15c51, r15c60 are toggled
        
        # Identify the toggle columns based on the pattern
        toggle_cols = [9, 18, 51, 60]
        toggle_rows = range(15, 24)
        
        for r in toggle_rows:
            for c in toggle_cols:
                # Toggle the cell
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 9
                else:
                    new_grid[r, c] = 5
                # Also toggle adjacent cells in a 3x3 pattern
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if new_grid[nr, nc] == 5:
                                new_grid[nr, nc] = 9
                            else:
                                new_grid[nr, nc] = 5
                                
    elif action == 5:
        # Action 5: Toggle specific rows (15-23) at specific columns
        # Based on observed deltas, this action toggles rows 15-23 at columns 6, 15, 45, 54
        # The pattern is: toggle 3x3 blocks or similar structures
        # From deltas: r15c6, r15c15, r15c45, r15c54 are toggled
        
        # Identify the toggle columns based on the pattern
        toggle_cols = [6, 15, 45, 54]
        toggle_rows = range(15, 24)
        
        for r in toggle_rows:
            for c in toggle_cols:
                # Toggle the cell
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 9
                else:
                    new_grid[r, c] = 5
                # Also toggle adjacent cells in a 3x3 pattern
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if new_grid[nr, nc] == 5:
                                new_grid[nr, nc] = 9
                            else:
                                new_grid[nr, nc] = 5
                                
    elif action == 6:
        # Action 6: Click at specific pixel coordinates
        if data and 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            # Convert pixel to logical coordinates
            r, c = py // 1, px // 1
            if 0 <= r < H and 0 <= c < W:
                # Toggle the cell
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 9
                else:
                    new_grid[r, c] = 5
                # Also toggle adjacent cells in a 3x3 pattern
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if new_grid[nr, nc] == 5:
                                new_grid[nr, nc] = 9
                            else:
                                new_grid[nr, nc] = 5
                                
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    
    # Check if the grid matches the win state pattern
    # The win state has specific patterns in rows 18-23 and 27-29
    
    # Check rows 18-23: should have pattern 9x21, 4x9, 9x6, 10x3, 9x6, 5x9, 9x9, 11x1
    for r in range(18, 24):
        row_str = ','.join([f"{grid[r, c]}x{1}" for c in range(W)])
        expected = "9x21,4x9,9x6,10x3,9x6,5x9,9x9,11x1"
        if row_str != expected:
            return False
            
    # Check rows 27-29: should have pattern 9x15, 4x9, 9x12, 10x3, 9x12, 5x9, 9x3, 11x1
    for r in range(27, 30):
        row_str = ','.join([f"{grid[r, c]}x{1}" for c in range(W)])
        expected = "9x15,4x9,9x12,10x3,9x12,5x9,9x3,11x1"
        if row_str != expected:
            return False
            
    # Check rows 0-17: should have pattern 9x36, 10x3, 9x24, 11x1
    for r in range(0, 18):
        row_str = ','.join([f"{grid[r, c]}x{1}" for c in range(W)])
        expected = "9x36,10x3,9x24,11x1"
        if row_str != expected:
            return False
            
    # Check rows 30-62: should have pattern 9x36, 10x3, 9x24, 11x1
    for r in range(30, 63):
        row_str = ','.join([f"{grid[r, c]}x{1}" for c in range(W)])
        expected = "9x36,10x3,9x24,11x1"
        if row_str != expected:
            return False
            
    return True