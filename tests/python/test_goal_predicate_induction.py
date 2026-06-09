import pytest
import numpy as np
from carnot.agentic.arc_agi3_goal_induction import induce_goal_predicate

def test_induce_goal_predicate_needs_two_examples():
    """REQ-PHASE4-009: Require >=2 level-up examples before committing"""
    win_grids = [np.zeros((5, 5))]
    non_win_grids = [np.ones((5, 5))]
    pred = induce_goal_predicate(win_grids, non_win_grids)
    assert pred is None

def test_induce_goal_predicate_object_reduction():
    """SCENARIO-PHASE4-009: Induces goal when win states have fewer objects"""
    # 2 win grids with 1 object each
    win_grids = [
        np.array([[0, 1], [0, 0]]),
        np.array([[0, 0], [1, 0]])
    ]
    # 2 non-win grids with 2 objects each
    non_win_grids = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 2]])
    ]
    
    pred = induce_goal_predicate(win_grids, non_win_grids)
    assert pred is not None
    
    # Should flag win grids
    for g in win_grids:
        assert pred(g) is True
        
    # Should flag non-win grids as False
    for g in non_win_grids:
        assert pred(g) is False

def test_induce_goal_predicate_color_reduction():
    """SCENARIO-PHASE4-009: Induces goal when win states have fewer colors"""
    # Win grids have only colors 0 and 1
    win_grids = [
        np.array([[0, 1, 1], [0, 0, 1]]),
        np.array([[1, 1, 0], [1, 0, 0]])
    ]
    # Non-win grids have colors 0, 1, and 2
    non_win_grids = [
        np.array([[0, 1, 2], [0, 0, 1]]),
        np.array([[1, 2, 0], [1, 0, 0]])
    ]
    
    pred = induce_goal_predicate(win_grids, non_win_grids)
    assert pred is not None
    
    for g in win_grids:
        assert pred(g) is True
        
    for g in non_win_grids:
        assert pred(g) is False

def test_induce_goal_predicate_missing_color():
    """SCENARIO-PHASE4-009: Induces goal when specific colors disappear"""
    # Non-win grids have color 3 that disappears in win grids
    non_win_grids = [
        np.array([[0, 3], [1, 2]]),
        np.array([[0, 3], [1, 2]])
    ]
    win_grids = [
        np.array([[0, 4], [1, 2]]),
        np.array([[0, 4], [1, 2]])
    ]
    # Object counts are the same (4 objects for each, since colors are distinct)
    # Colors count is the same (4 colors each)
    # But color '3' disappears!
    pred = induce_goal_predicate(win_grids, non_win_grids)
    assert pred is not None
    
    for g in win_grids:
        assert pred(g) is True
    for g in non_win_grids:
        assert pred(g) is False

def test_induce_goal_predicate_fallback():
    """SCENARIO-PHASE4-009: Induces fallback object count goal"""
    # Create grids such that max_win_objs is NOT < min_non_win_objs
    # and colors are the same, and no common colors disappear.
    win_grids = [
        np.array([[0, 1], [0, 0]]), # 1 obj
        np.array([[0, 1], [0, 1]])  # 2 objs
    ]
    non_win_grids = [
        np.array([[0, 1], [0, 1]]), # 2 objs
        np.array([[0, 0], [0, 0]])  # 0 objs
    ]
    # Here max_win_objs = 2, min_non_win_objs = 0.
    # Hypothesis 1 fails. Hypothesis 2 fails (win_objs not same).
    # Hypothesis 3 fails (max_win_colors=2, min_non_win_colors=1).
    # Hypothesis 4 fails (disappearing colors empty).
    pred = induce_goal_predicate(win_grids, non_win_grids)
    assert pred is not None
    
    # Fallback returns lambda g: len(objects(g)) <= max_win_objs (which is 2)
    # Win grids have <= 2 objects, so they return True
    assert pred(win_grids[0]) is True
    assert pred(win_grids[1]) is True
