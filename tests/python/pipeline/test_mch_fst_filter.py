import math
from carnot.pipeline.mch_fst_filter import MCHFSTFilter

def test_mch_fst_filter_acceptance_rate():
    # Mock energy function: energy increases with length
    def dummy_energy(text):
        return float(len(text))

    filter_obj = MCHFSTFilter(dummy_energy)
    
    # 1. Test accept prob
    assert filter_obj.accept_prob(1.0, 1.0) == 1.0
    assert filter_obj.accept_prob(1.0, 0.0) == 1.0
    assert math.isclose(filter_obj.accept_prob(1.0, 2.0), math.exp(-1.0))
    
    # 2. Test token filtering
    prompt = "A"
    tokens = ["B", "C", "D"]
    # B increases energy by 1 => prob = 1/e
    # if random state returns 0.1 (< 1/e ~ 0.36), it accepts B
    # let's mock random to always accept
    class MockRandom:
        def random(self):
            return 0.0  # Always accept
    
    accepted = filter_obj.filter_tokens(prompt, tokens, random_state=MockRandom())
    assert accepted == ["B", "C", "D"]
    
    # let's mock random to always reject if prob < 1
    class MockRandomReject:
        def random(self):
            return 0.99  # > 1/e, so rejects energy increases
            
    accepted_reject = filter_obj.filter_tokens(prompt, tokens, random_state=MockRandomReject())
    assert accepted_reject == []

def test_mch_fst_filter_decrease_energy():
    # Energy decreases, so prob=1.0, should always accept regardless of random
    def decreasing_energy(text):
        return float(100 - len(text))
        
    filter_obj = MCHFSTFilter(decreasing_energy)
    
    class MockRandomReject:
        def random(self):
            return 0.99
            
    prompt = "A"
    tokens = ["B", "C"]
    
    accepted = filter_obj.filter_tokens(prompt, tokens, random_state=MockRandomReject())
    assert accepted == ["B", "C"]
