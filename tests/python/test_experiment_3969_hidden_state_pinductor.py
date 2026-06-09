import pytest
import numpy as np
from carnot.agentic.arc_pinductor import PinductorModel
from carnot.agentic.arc_pinductor_candidates import get_candidates

def dummy_traj():
    s0 = np.zeros((2,2), dtype=int)
    s1 = np.ones((2,2), dtype=int)
    a = (6, 0, 0)
    return [
        (s0, a, s0),
        (s0, a, s1),
        (s1, a, s1),
        (s1, a, s0)
    ]

def step_fn(L, s, a):
    return (L + 1) % 2

def test_pinductor_perfect_prediction():
    model = PinductorModel("test", step_fn, 2)
    traj = dummy_traj()
    model.fit([traj])
    res = model.consistency_energy([traj])
    assert res["energy"] == 0.0
    assert res["transition_exact_rate"] == 1.0

def test_pinductor_fallback_belief():
    model = PinductorModel("test", step_fn, 2)
    traj = dummy_traj()
    model.fit([traj])
    
    belief = {0: 1.0}
    # Unseen state
    s_unseen = np.full((2,2), 2, dtype=int)
    a = (6, 0, 0)
    pred = model.predict_belief(s_unseen, a, belief)
    assert np.array_equal(pred, s_unseen)
    
    # Empty belief
    pred = model.predict_belief(np.zeros((2,2), dtype=int), a, {})
    assert np.array_equal(pred, np.zeros((2,2), dtype=int))
    
    # Update with unseen state
    belief = model.update_belief(s_unseen, a, s_unseen, belief)
    assert isinstance(belief, dict)

def test_pinductor_empty_heldout():
    model = PinductorModel("test", step_fn, 2)
    res = model.consistency_energy([])
    assert res["energy"] is None

def test_pinductor_changed_but_wrong_shape():
    model = PinductorModel("test", step_fn, 2)
    traj = dummy_traj()
    model.fit([traj])
    # Different shape target
    s0 = np.zeros((2,2), dtype=int)
    s_diff = np.zeros((3,3), dtype=int)
    a = (6, 0, 0)
    res = model.consistency_energy([[(s0, a, s_diff)]])
    assert res["energy"] is None

def test_pinductor_candidates():
    cands = get_candidates()
    assert len(cands) > 0
    # ensure structure
    for name, fn, K in cands:
        assert isinstance(name, str)
        assert callable(fn)
        assert isinstance(K, int)

    # Test logic
    s = np.zeros((2,2), dtype=int)
    s[0, 0] = 5
    a_click = (6, 0, 0)
    a_click_other = (6, 1, 1)
    a_kbd = (2,)
    
    by_name = {c[0]: c[1] for c in cands}
    
    # Step mod
    assert by_name["step_mod_2"](0, s, a_click) == 1
    
    # Color click
    assert by_name["color_click_5_mod_2"](0, s, a_click) == 1
    assert by_name["color_click_5_mod_2"](0, s, a_click_other) == 0
    assert by_name["color_click_5_mod_2"](0, s, a_kbd) == 0
    
    # Any click
    assert by_name["any_click_mod_2"](0, s, a_click) == 1
    assert by_name["any_click_mod_2"](0, s, a_kbd) == 0
    
    # Action type
    assert by_name["action_type_2_mod_2"](0, s, a_kbd) == 1
    assert by_name["action_type_2_mod_2"](0, s, a_click) == 0

def test_pinductor_update_belief_zero_prob():
    model = PinductorModel("test", step_fn, 2)
    s0 = np.zeros((2,2), dtype=int)
    a = (6, 0, 0)
    # fit nothing
    belief = {0: 1.0}
    b = model.update_belief(s0, a, s0, belief)
    assert 0 in b
