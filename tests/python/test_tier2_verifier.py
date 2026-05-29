import pytest
import math
from carnot.cascade.tier2_verifier import PrefixClosedBoundVerifier, TrieNode

def test_trie_node():
    node = TrieNode(1, 0.5)
    assert node.token_id == 1
    assert node.prob == 0.5
    assert not node.is_terminal
    assert not node.is_pruned

def test_verifier_add_expansion_and_bounds():
    verifier = PrefixClosedBoundVerifier()
    
    # Root starts with lower=0, upper=1
    lower, upper = verifier.compute_bounds()
    assert math.isclose(lower, 0.0)
    assert math.isclose(upper, 1.0)
    
    # Expand root
    # Token 1 (prob 0.6), Token 2 (prob 0.4)
    verifier.add_expansion((), {1: 0.6, 2: 0.4}, is_violation=False)
    
    # Prune token 2 (is_violation=True)
    verifier.add_expansion((2,), {}, is_violation=True)
    lower, upper = verifier.compute_bounds()
    assert math.isclose(lower, 0.0)
    assert math.isclose(upper, 0.6) # 1 - 0.4
    
    # Add expansion for token 1 -> token 3 (0.5), token 4 (0.5)
    verifier.add_expansion((1,), {3: 0.5, 4: 0.5}, is_violation=False)
    
    # Token 1, 3 is terminal success
    verifier.add_expansion((1, 3), {}, is_violation=False, is_terminal=True)
    lower, upper = verifier.compute_bounds()
    assert math.isclose(lower, 0.3) # 0.6 * 0.5
    assert math.isclose(upper, 0.6)
    
    # Check monotonicity
    assert verifier.check_monotonicity((0.0, 1.0))
    assert not verifier.check_monotonicity((0.4, 1.0)) # lower went down (0.3 < 0.4) -> False
    
    # Prune an already pruned node should do nothing
    verifier.add_expansion((2,), {}, is_violation=True)
    lower2, upper2 = verifier.compute_bounds()
    assert math.isclose(lower, lower2)
    assert math.isclose(upper, upper2)
    
    # Expand a node that wasn't previously added, auto-creating path
    verifier.add_expansion((1, 4, 5), {6: 1.0}, is_violation=False)
    assert verifier._get_node((1, 4, 5)) is not None
    assert verifier._get_node((1, 4, 5, 6)) is not None
    assert 6 in verifier._get_node((1, 4, 5)).children
    
def test_sample_estimate():
    verifier = PrefixClosedBoundVerifier()
    
    # A dummy evaluate function that returns True 40% of the time based on an internal counter
    counter = [0]
    def eval_fn():
        counter[0] += 1
        return (counter[0] % 10) < 4
        
    est1, est2 = verifier.sample_estimate(100, eval_fn)
    assert est1 == est2
    assert math.isclose(est1, 0.4)
