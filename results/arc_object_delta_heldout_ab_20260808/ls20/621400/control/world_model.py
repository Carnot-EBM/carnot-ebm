import numpy as np

def engine(grid, action, data):
    return np.array(grid, copy=True)

def is_level_complete(grid):
    return False

ENGINE_RECEIPT = {"arm":"control","game":"ls20","metric":{"action_budget":0,"change_fidelity":0.02374,"engine_loaded":true,"engine_source":"matched_fixture_engine","goal_fidelity":0.51187,"prompt_chars":13450,"wall_s":0.5345}}
