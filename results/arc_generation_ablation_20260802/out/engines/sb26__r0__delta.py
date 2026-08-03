import numpy as np

def engine(grid, action, data=None):
    if action != 6:
        return grid
    
    y, x = data['y'], data['x']
    out = grid.copy()
    
    # The game seems to be about clicking on "buttons" (the colored blocks at the bottom)
    # and seeing them affect "slots" in the middle area.
    # Based on the observed transitions:
    # Click (36, 59) -> affects r56c33-38, r61c33-38 etc.
    # Click (20, 59) -> affects r56c17-22, r60c17-22 etc.
    # Click (44, 59) -> affects r56c41-46, r60c41-46 etc.
    # These are buttons at the bottom (r57-60).
    # Buttons are located at c18-21 (color 14), c26-29 (color 15), c34-37 (color 9), c42-46 (color 11).
    # Wait, looking at INITIAL GRID:
    # r57-60: c18-21(14), c26-29(15), c34-37(9), c42-45(11).
    # Let's map them to colors:
    # Button Color 14: x=18..21, y=57..60
    # Button Color 15: x=26..29, y=57..60
    # Button Color 9:  x=34..37, y=57..60
    # Button Color 11: x=42..45, y=57..60
    
    # Now let's look at ACTION6 data={'x': 36, 'y': 59} -> changes cells in the button area.
    # It seems clicking a button "activates" it or "toggles" it.
    # Clicking (36, 59) is Button 9.
    # Clicking (20, 59) is Button 14.
    # Clicking (44, 59) is Button 11.
    
    # The transitions show that when you click a button, it might change color to 0 (empty).
    # But then subsequent clicks on slots (r28-31, c21-45) also trigger these buttons.
    # Let's refine the mapping:
    # Buttons are at r57-60 and columns:
    # Col range [18, 21] -> Color 14
    # Col range [26, 29] -> Color 15
    # Col range [34, 37] -> Color 9
    # Col range [42, 45] -> Color 11
    
    # Mapping from x coordinate of click to which button/color:
    # If x in [18, 21], btn_color = 14
    # If x in [26, 29], btn_color = 15
    # If x in [34, 37], btn_color = 9
    # If x in [34, 37], wait... let me re-check.
    # Initial grid r57: "4x18, 14x4, 4x4, 15x4, 4x4, 9x4, 4x4, 11x4, 4x18"
    # c0-17: 4, c18-21: 14, c22-25: 4, c26-29: 15, c30-33: 4, c34-37: 9, c38-41: 4, c42-45: 11, c46-63: 4
    
    # Now look at the transitions again.
    # Click (36, 59) -> Button 9. Changes cells to 0.
    # Click (20, 59) -> Button 14. Changes cells to 0.
    # Click (44, 59) -> Button 11. Changes cells to 0.
    # These clicks on buttons themselves just clear them?
    
    # Let's see what happens when clicking a slot in the middle (r28-31).
    # Click (23, 30) -> Slot [c21-24]. Result: r28-31, c21-24 becomes color 9.
    # Also affects button area: r56c33-38 becomes 4.
    # Wait, if you click a slot, it fills with a color and some button is restored/changed.
    
    # The slots are:
    # Slot 1: c21-24
    # Slot 2: c27-30
    # Slot 3: c33-36
    # Slot 4: c37-40... no.
    # Looking at ACTION6 data={'x': 23, 'y': 30}: x=23 is in Slot 1.
    # It fills Slot 1 (r28-31, c21-24) with color 9.
    # And it restores Button 9 (r56c33-38 etc)? No, let's look at delta: "r56c33:0x6" was the first action.
    # Then "r56c33:4x6" happens when clicking Slot 1.
    # This means Clicking Slot 1 -> Fills Slot 1 with Color 9 AND Restores Button 9.
    # Let's check others:
    # Click (29, 30) -> Slot [c27-30]. Result: r28-31, c27-30 becomes color 14.
    # Also restores Button 14 (r56c17-22).
    # Click (35, 30) -> Slot [c33-36]. Result: r28-31, c33-36 becomes color 11.
    # Wait, this is confusing. Let's map carefully.
    
    # Slots and their associated colors/buttons:
    # Slot 1 (x=21..24): fills with Color 9, restores Button 9 (x=34..37)
    # Slot 2 (x=27..30): fills with Color 14, restores Button 14 (x=18..21)
    # Slot 3 (x=33..36): fills with Color 11, restores Button 11 (x=42..45)
    
    # This doesn't seem to be a simple mapping. Let's look at the coordinates again.
    # x=23 (Slot 1) -> Color 9, Button 9 restored.
    # x=29 (Slot 2) -> Color 14, Button 14 restored.
    # x=35 (Slot 3) -> Color 11, Button 11 restored.
    
    # Mapping:
    # Click x in [21, 24] -> Fill r28-31, c21-24 with color 9; restore button at x=34-37.
    # Click x in [27, 30] -> Fill r28-31, c27-30 with color 14; restore button at x=18-21.
    # Click x in [33, 36] -> Fill r28-31, c33-36 with color 11; restore button at x=18-21? No, let me re-read.
    # "r56c41:4x6" is for click (35, 30). That's Button 11.
    
    # Correct mapping:
    # Slot 1 (x=21..24) -> fills with 9, restores Button 9 (x=34..37)
    # Slot 2 (x=27..30) -> fills with 14, restores Button 14 (x=18..21)
    # Slot 3 (x=33..36) -> fills with 11, restores Button 11 (x=42..45)
    
    # Now what about the buttons themselves?
    # Click (36, 59) -> Button 9. Result: r56c33-38 becomes 0.
    # Click (20, 59) -> Button 14. Result: r56c17-22 becomes 0.
    # Click (44, 59) -> Button 11. Result: r56c41-46 becomes 0.
    
    # Let's implement this logic.
    if y >= 57 and y <= 60:
        if x >= 18 and x <= 21: # Button 14
            out[56:62, 17:23] = 0 if out[56, 17] != 0 else 4
            return out
        elif x >= 26 and x <= 29: # Button 15
            # No observed transition for button 15, but follow pattern
            out[56:62, 25:31] = 0 if out[56, 25] != 0 else 4
            return out
        elif x >= 34 and x <= 37: # Button 9
            out[56:62, 33:39] = 0 if out[56, 33] != 0 else 4
            return out
        elif x >= 42 and x <= 45: # Button 11
            out[56:62, 41:47] = 0 if out[56, 41] != 0 else 4
            return out
    
    if y >= 28 and y <= 31:
        if x >= 21 and x <= 24: # Slot 1
            out[28:32, 21:25] = 9
            out[56:62, 33:39] = 4
            return out
        elif x >= 27 and x <= 30: # Slot 2
            out[28:32, 27:31] = 14
            out[56:62, 17:23] = 4
            return out
        elif x >= 33 and x <= 36: # Slot 3
            out[28:32, 33:37] = 11
            out[56:62, 41:47] = 4
            return out
            
    return out

def is_level_complete(grid):
    # The win state isn't provided, but typically it's when slots are filled.
    # Check if the three slots we saw being filled are now color 9, 14, 11.
    return np.all(grid[28:32, 21:25] == 9) and \
           np.all(grid[28:32, 27:31] == 14) and \
           np.all(grid[28:32, 33:37] == 11)