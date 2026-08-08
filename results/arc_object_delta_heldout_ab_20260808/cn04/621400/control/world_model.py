import numpy as np

def engine(grid, action, data):
    return np.array(grid, copy=True)

def is_level_complete(grid):
    return False

ENGINE_RECEIPT = {"arm":"control","game":"cn04","metric":{"action_budget":0,"change_fidelity":0.02315,"engine_loaded":true,"engine_source":"matched_fixture_engine","goal_fidelity":0.511575,"prompt_chars":14419,"wall_s":0.54419}}
