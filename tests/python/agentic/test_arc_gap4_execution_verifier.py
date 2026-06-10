import pytest
import numpy as np
from carnot.agentic.arc_gap4_execution_verifier import Gap4ExecutionVerifier, extract_dsl_rules, apply_rule, get_consistency_energy

def test_extract_dsl_rules_empty():
    assert extract_dsl_rules([]) is None
    assert extract_dsl_rules(None) is None

def test_extract_dsl_rules_identity():
    demos = [
        {'input': [[1, 2], [3, 4]], 'output': [[1, 2], [3, 4]]}
    ]
    rule = extract_dsl_rules(demos)
    assert rule == ('identity',)

def test_extract_dsl_rules_recolor():
    demos = [
        {'input': [[1, 2], [3, 4]], 'output': [[5, 2], [3, 4]]},
        {'input': [[1, 1], [3, 4]], 'output': [[5, 5], [3, 4]]}
    ]
    rule = extract_dsl_rules(demos)
    assert rule == ('recolor', 1, 5)

def test_extract_dsl_rules_shape_mismatch():
    demos = [
        {'input': [[1, 2]], 'output': [[1, 2, 3]]}
    ]
    assert extract_dsl_rules(demos) is None

def test_extract_dsl_rules_no_change_one_demo():
    # first demo changes, second demo doesn't (so recolor logic triggers continue on diff)
    demos = [
        {'input': [[1, 2], [3, 4]], 'output': [[5, 2], [3, 4]]},
        {'input': [[2, 2], [3, 4]], 'output': [[2, 2], [3, 4]]}
    ]
    assert extract_dsl_rules(demos) == ('recolor', 1, 5)

def test_extract_dsl_rules_multiple_color_changes():
    demos = [
        {'input': [[1, 2]], 'output': [[5, 6]]}
    ]
    assert extract_dsl_rules(demos) is None

def test_extract_dsl_rules_not_valid_for_all():
    demos = [
        {'input': [[1, 2]], 'output': [[5, 2]]}, # 1 -> 5
        {'input': [[1, 2]], 'output': [[1, 2]]}  # 1 -> 1 (not 5)
    ]
    assert extract_dsl_rules(demos) is None

def test_apply_rule():
    grid = [[1, 2], [3, 4]]
    assert np.array_equal(apply_rule(('identity',), grid), grid)
    assert np.array_equal(apply_rule(('recolor', 1, 5), grid), [[5, 2], [3, 4]])
    assert apply_rule(None, grid) is None
    assert apply_rule(('unknown',), grid) is None

def test_get_consistency_energy():
    rule = ('recolor', 1, 5)
    test_input = [[1, 2], [3, 4]]
    gold_candidate = [[5, 2], [3, 4]]
    near_miss = [[5, 2], [3, 9]]
    
    assert get_consistency_energy(rule, test_input, gold_candidate) == 0.0
    assert get_consistency_energy(rule, test_input, near_miss) == 0.25
    
    # rule is none
    assert get_consistency_energy(None, test_input, gold_candidate) == 1.0
    
    # shape mismatch
    assert get_consistency_energy(rule, test_input, [[5, 2]]) == 1.0

def test_verifier_class():
    verifier = Gap4ExecutionVerifier()
    
    # success
    demos = [{'input': [[1]], 'output': [[2]]}]
    rule = verifier.induce_program(demos)
    assert rule == ('recolor', 1, 2)
    assert verifier.llm_proposer_used == False
    
    # fail (returns None)
    demos_fail = [{'input': [[1, 2]], 'output': [[5, 6]]}]
    assert verifier.induce_program(demos_fail) is None
