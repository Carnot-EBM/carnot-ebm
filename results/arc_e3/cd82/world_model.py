import numpy as np

def engine(grid, action, data):
    """
    grid: np.ndarray (logical 64x64 int).
    action: int (1-7).
    data: dict (for action 6).
    Returns: np.ndarray (predicted next grid).
    """
    new_grid = grid.copy()
    
    # The observed transitions show a complex "filling" or "expansion" 
    # mechanism where certain values (2, 5, 15) interact.
    # Specifically, Action 2 and 4 seem to trigger a "growth" or 
    # "filling" of a region based on the current state.
    # Action 3 and 5 seem to be "toggling" or "state-changing" 
    # operations on specific regions.
    
    # Since we must produce a deterministic engine based on the 
    # observed deltas, and the deltas are provided as a list of 
    # (row, col, from, to), we simulate the logic of the deltas.
    
    # However, the prompt asks to induce the transition RULES.
    # Looking at the deltas:
    # Action 2: Expansion of '5' into '2' and '15' into '5'.
    # Action 3: Expansion of '2' into '5' and '15' into '2'.
    # Action 4: Expansion of '2' into '5' and '15' into '5'.
    # Action 5: A large-scale state change (filling 0s with 15s, 
    #             changing 3s to 4s, etc.)
    
    # Because the deltas are very specific, we implement the 
    # logic that produces those deltas.
    
    # Action 2 Logic:
    if action == 2:
        # Expansion of 2s and 15s into 5s
        # This is a "filling" rule.
        for r in range(64):
            for c in range(64):
                if new_grid[r, c] == 15: