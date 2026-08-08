import numpy as np

def engine(grid, action, data):
    return np.array(grid, copy=True)

def is_level_complete(grid):
    return False

ENGINE_RECEIPT = {"arm":"aa_control_b","game":"s5i5","metric":{"action_budget":0,"change_fidelity":0.02311,"engine_loaded":true,"engine_source":"matched_fixture_engine","goal_fidelity":0.511555,"prompt_chars":14304,"wall_s":0.54304}}
