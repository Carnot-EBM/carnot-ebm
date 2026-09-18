import numpy as np
def engine(grid, action, data):
    return grid + 1
def is_level_complete(grid):
    return bool(np.all(grid >= 1))
