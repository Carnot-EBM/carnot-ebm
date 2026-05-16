import pytest
from carnot.pipeline.fast_slow_variant import VerifyResult
from carnot.pipeline.cot2meta_routing import cot2meta_state_machine, evaluate_odar_action

def test_evaluate_odar_action():
    # REQ-COT2META-001
    action = evaluate_odar_action(0.1, 0.2, 0.3, 0.4, 0.5, risk_sensitivity=2.0)
    assert action == "fallback"
    
def test_cot2meta_stop():
    # SCENARIO-COT2META-STOP
    def base_llm(p): return "answer"
    def verifier(r): return VerifyResult(False, "err")
    def odar(r, b, p): return "stop"
    
    res = cot2meta_state_machine("p", base_llm, verifier, odar, max_iters=5)
    assert res["iters"] == 1
    assert res["final_action"] == "stop"
    assert not res["fallback_triggered"]
    assert not res["passed"]

def test_cot2meta_fallback():
    # SCENARIO-COT2META-FALLBACK
    def base_llm(p): return "answer"
    def verifier(r): return VerifyResult(False, "err")
    def odar(r, b, p): return "fallback"
    
    res = cot2meta_state_machine("p", base_llm, verifier, odar, max_iters=5)
    assert res["iters"] == 1
    assert res["final_action"] == "fallback"
    assert res["fallback_triggered"]
    assert not res["passed"]

def test_cot2meta_prune_expand_repair_pass():
    # SCENARIO-COT2META-PASS
    actions = ["prune", "expand", "repair"]
    def base_llm(p): return "answer"
    call_count = 0
    def verifier(r):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            return VerifyResult(True, "")
        return VerifyResult(False, "err")
        
    def odar(r, b, p):
        return actions[len(b) % len(actions)]
    
    res = cot2meta_state_machine("p", base_llm, verifier, odar, max_iters=5)
    assert res["passed"]
    assert res["iters"] == 3
    assert not res["fallback_triggered"]
    assert res["final_action"] in actions
