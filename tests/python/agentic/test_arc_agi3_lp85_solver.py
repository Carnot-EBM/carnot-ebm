import numpy as np
import pytest
from carnot.agentic.arc_agi3_lp85_solver import (
    grid_of,
    compute_grid_delta,
    objects,
    discover_buttons,
    plan_bfs,
    attempt_solve,
)

class DummyFrame:
    def __init__(self, arr, levels_completed=0):
        self.frame = arr
        self.levels_completed = levels_completed

class DummyGame:
    def __init__(self, levels_completed=0):
        self._levels_completed = levels_completed
        
class DummyEnv:
    def __init__(self):
        self._game = DummyGame()
        self.step_count = 0
        
    def reset(self):
        self.step_count = 0
        self._game._levels_completed = 0
        return DummyFrame(np.zeros((10, 10)), levels_completed=0)
        
    def step(self, action, data):
        self.step_count += 1
        arr = np.zeros((10, 10))
        if data.get("x") == 4 and data.get("y") == 4:
            arr[1, 1] = 1 # simulated change
        if self.step_count >= 2:
            self._game._levels_completed = 1
        return DummyFrame(arr, levels_completed=self._game._levels_completed)

def test_grid_of():
    f = DummyFrame(np.array([[[1, 2], [3, 4]]]))
    g = grid_of(f)
    assert g.shape == (2, 2)
    assert g[0, 0] == 1

def test_compute_grid_delta():
    g1 = np.zeros((2, 2))
    g2 = np.ones((2, 2))
    delta = compute_grid_delta(g1, g2)
    assert delta["n_changed"] == 4
    
    g3 = np.zeros((3, 3))
    delta2 = compute_grid_delta(g1, g3)
    assert delta2["n_changed"] == -1

def test_objects():
    g = np.zeros((10, 10))
    g[2:4, 2:4] = 1 # object 1
    g[6:8, 6:8] = 2 # object 2
    objs = objects(g)
    assert len(objs) == 2
    # centroids
    assert (2, 2) in objs or (3, 3) in objs or (2, 3) in objs or (3, 2) in objs

def test_discover_buttons():
    env = DummyEnv()
    start_grid = np.zeros((10, 10))
    # inject an object at (4,4)
    start_grid[4, 4] = 1
    
    buttons = discover_buttons(env, start_grid)
    assert buttons == [(4, 4)]

def test_plan_bfs():
    env = DummyEnv()
    start_grid = np.zeros((10, 10))
    start_grid[4, 4] = 1
    buttons = [(4, 4)]
    
    path = plan_bfs(env, start_grid, buttons, start_levels=0, max_depth=5)
    assert path == [(4, 4), (4, 4)]

def test_attempt_solve():
    env = DummyEnv()
    # we need objects to return [(4,4)] when called on reset
    # override objects in solver just for this test?
    # Better: just use a patch
    import carnot.agentic.arc_agi3_lp85_solver as solver
    old_objects = solver.objects
    solver.objects = lambda g: [(4, 4)]
    
    f, actions, log = attempt_solve(env, budget=5)
    assert actions == 1
    assert f.levels_completed == 1
    assert len(log) == 1
    
    # Restore
    solver.objects = old_objects
