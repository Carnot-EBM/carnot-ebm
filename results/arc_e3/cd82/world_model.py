import numpy as np

def engine(grid, action, data):
    # Grid is 8x8. Action 1-7 are directional.
    # Action 6 is a click at pixel coordinates (data={'x':px, 'y':py}).
    # Based on the provided transitions (empty), we assume a default behavior
    # or that the grid remains unchanged if no specific logic is triggered.
    # However, usually ARC tasks imply movement or interaction.
    # Since no transitions were provided to infer logic, we return the grid as is.
    # This is a placeholder for the actual logic which would be derived from transitions.
    # Given the strict instruction to output ONLY the code block and the lack of transitions,
    # we must assume the simplest possible model or that the prompt implies a specific known task.
    # Without transitions, we cannot infer movement or gravity.
    # We will implement a basic movement logic for actions 1-7 (Up, Down, Left, Right, etc.)
    # assuming the player is at a specific location or the grid represents a static world.
    # Since the prompt asks for a world model and provides no transitions, we return the grid.
    return grid.copy()

def is_level_complete(grid):
    # Without transitions, we cannot define a win state.
    # We return False as a default.
    return False