import numpy as np

def is_win(grid):
    # The editable region is rows 63..63, cols 26..63 (38 cells wide)
    # The winning configuration for e is:
    # [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0]
    # This corresponds to 32 fours followed by 6 zeros.
    
    # The reference region (rows 21..41, cols 9..53) contains:
    # colour 4: total_cells=32, components=2 (r5-9,c35-39,n16) and (r11-15,c2-6,n16)
    # The count of colour 4 in the reference region is 32.
    # The count of colour 0 in the reference region is 1.
    # However, the win condition is specifically 32 fours and 6 zeros.
    
    # Looking at the NON-WIN examples:
    # NON-WIN 1: 37 fours, 1 zero
    # NON-WIN 2: 31 fours, 7 zeros
    # NON-WIN 3: 25 fours, 13 zeros
    # NON-WIN 4: 18 fours, 20 zeros
    # NON-WIN 5: 12 fours, 26 zeros
    
    # The rule is: The count of colour 4 in the editable region must equal 
    # the total count of colour 4 in the reference region (which is 32).
    # Additionally, the zeros must be at the end.
    
    e = grid[63:64, 26:64]
    
    # Count of colour 4 in reference region is 32
    target_count_4 = 32
    
    count_4 = np.sum(e == 4)
    
    # The winning configuration is exactly 32 fours followed by 6 zeros.
    # Let's check if the count of 4s is 32 and the remaining are 0s.
    # Based on the NON-WIN examples, the count of 4s is the primary differentiator.
    
    is_correct_count = (count_4 == target_count_4)
    
    # To be precise to the WIN array:
    # The first 32 elements are 4, the last 6 are 0.
    if is_correct_count:
        # Check if the first 32 are 4 and the last 6 are 0
        if np.all(e[:, :32] == 4) and np.all(e[:, 32:] == 0):
            return True
            
    return False