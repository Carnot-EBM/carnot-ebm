import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state based on the current grid, action, and data.
    """
    # Based on the observed transition, Action 1 increments the cell at (0, 0) by 1.
    # The grid is 1x1.
    if action == 1:
        if grid[0, 0] == 0:
            grid[0, 0] = 1
        elif grid[0, 0] == 1:
            grid[0, 0] = 2
        # Assuming a cycle or increment logic, though the example only shows 0->1.
        # Given the simplicity, we assume a simple increment or toggle.
        # Let's assume a simple increment for now, but since it's a 1x1 grid,
        # we need to know the max value.
        # If it's a toggle, 0->1, 1->0.
        # If it's a counter, 0->1, 1->2.
        # Without more data, we can't be sure.
        # However, usually in ARC, 0 is empty, 1 is filled.
        # Let's assume a toggle for now as it's common.
        # Wait, the prompt says "induce rules".
        # Let's look at the delta: r0c0:1.
        # This means the cell became 1.
        # If the previous was 0, it became 1.
        # If the previous was 1, what happens?
        # Let's assume a simple increment for now.
        # But wait, if it's a toggle, 1->0.
        # Let's assume a toggle.
        grid[0, 0] = 1 - grid[0, 0]
    return grid

def is_level_complete(grid):
    """
    Checks if the grid represents a win state.
    """
    # Based on the observed transition, the level is complete if the grid is all 1s.
    # Or maybe if the grid is all 0s.
    # Given the action 1 makes it 1, maybe the goal is to fill it.
    # Let's assume the goal is to have