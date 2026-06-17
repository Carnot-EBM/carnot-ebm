import numpy as np

def engine(grid, action, data):
    """
    grid: np.ndarray (logical 64x64 int).
    action: int (1-7).
    data: dict (for action 6).
    Returns: Predicted next grid.
    """
    new_grid = grid.copy()
    
    # The observed transitions show a complex "filling" or "spreading" mechanic.
    # In ACTION2, 3, and 4, a region of '2's (liquid/sand) expands into '15's (empty/void)
    # or '5's (solid/wall) depending on the specific transition.
    # However, the prompt asks to induce the rules from the observed data.
    # Looking at the DELTAS:
    # ACTION2: 2 -> 5 (filling/solidifying)
    # ACTION3: 5 -> 2 (melting/liquefying) and 2 -> 15 (evaporating/voiding)
    # ACTION4: 2 -> 5 (solidifying) and 5 -> 15 (voiding)
    # ACTION5: A massive structural change (level completion logic).
    
    # Since we must be deterministic and the DELTAS are provided for specific frames,
    # but we need a general engine, we observe the "flow" of values.
    # The values 2, 5, 15, 0 are key.
    # 15 is often a "void" or "passable" area.
    # 5 is a "solid" or "wall".
    # 2 is a "liquid" or "active" state.
    
    # Based on the DELTAS provided in the prompt:
    # Action 2: 2 -> 5
    # Action 3: 5 -> 2, 2 -> 15
    # Action 4: 2 -> 5, 5 -> 15
    
    # Because the DELTAS are specific to the state, a general engine 
    # for this specific game 'cd82' would simulate the "flow" of 2s into 15s