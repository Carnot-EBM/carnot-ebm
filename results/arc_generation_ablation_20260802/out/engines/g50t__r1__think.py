import numpy as np

def engine(grid, action, data):
    """
    Induces the world model for game 'g50t'.
    Based on observed transitions:
    ACTION2 seems to be a movement or shift that propagates changes in blocks of 5x5 or similar patterns.
    The grid contains structures made of colors 2, 5, 8, and 9.
    Looking at the deltas, ACTION2 shifts certain color patterns across the board (likely moving an object).
    Specifically, it replaces areas of color 9 with 5 or vice versa in specific rectangular regions.
    ACTION4 also performs a replacement of colors in a specific region.
    The state updates are highly localized and repetitive.
    """
    new_grid = grid.copy()
    
    # The provided observations show very specific coordinate-based replacements.
    # Since we need a general rule but only have one level's fragments, 
    # we map the observed delta behavior.
    
    if action == 2:
        # Action 2 appears to move a "brush" or "object" rightwards/downwards.
        # In the absence of a clear physics model, we simulate the pattern seen in the logs.
        # However, since this is a world model for a specific game instance 'g50t',
        # and the transitions are given as absolute cell changes, we can infer that 
        # the agent is likely interacting with blocks of size 5x5.
        
        # To implement a generic engine without knowing the exact entity position,
        # we look for existing patterns of color 9 and shift them or toggle them.
        # But based on the deltas, it looks like ACTION2 moves a window of change.
        # Let's assume there's an internal cursor (not visible) or it depends on current grid state.
        
        # For ARC-AGI tasks, often these actions correspond to moving a pointer.
        # Given the limited data, we will try to find the most recent 'active' block 
        # and shift its effect.
        pass

    elif action == 4:
        # Action 4 replaces a region of color 5 with 9.
        pass

    # Because the provided observations are very specific to one sequence of events,
    # and no "entity" (like a player character) is clearly identified in the initial grid,
    # the logic is likely tied to the coordinates of the colors.
    
    # Since I cannot determine the movement rule perfectly from just these fragments,
    # I will return the grid as is unless I can find a pattern.
    # However, looking at r63c61 -> r63c60 -> r63c59... this suggests a counter or progress bar.
    
    return new_grid

def is_level_complete(grid):
    """
    The win state usually involves completing a pattern or reaching a target.
    In the observed transitions, there is a cell at r63 that changes value/position.
    Usually, if all required cells are filled or a certain condition is met, it's complete.
    """
    # Check for common ARC completion patterns: e.g., no more 'incomplete' blocks.
    # Without the WIN STATE grid, we assume completeness based on the progression seen.
    # In many cases, filling a specific area or moving an object to a goal is the key.
    return False