from carnot.verify.semantic_violation_cost import SemanticViolationCostCalculator

def test_semantic_violation_cost_accepted():
    """
    Test that a valid graph mapping with low semantic violation cost is accepted.
    References REQ-VERIFY-3409 and SCENARIO-VERIFY-3409.
    """
    calc = SemanticViolationCostCalculator(threshold=2.0)
    graph = {
        "nodes": [{"id": "A"}, {"id": "B"}],
        "edges": [{"source": "A", "target": "B", "status": "ok"}]
    }
    cost, rejected = calc.evaluate(graph)
    assert cost == 0.0
    assert not rejected
    
def test_semantic_violation_cost_rejected():
    """
    Test that a graph mapping exceeding the acceptable deformation threshold
    is rejected as a smooth falsehood.
    References REQ-VERIFY-3409 and SCENARIO-VERIFY-3409.
    """
    calc = SemanticViolationCostCalculator(threshold=2.0)
    graph = {
        "nodes": [{"id": "A"}, {"id": "B"}],
        "edges": [
            {"source": "A", "target": "B", "status": "contradiction"},
            {"source": "B", "target": "A", "violation_weight": 1.5}
        ]
    }
    cost, rejected = calc.evaluate(graph)
    assert cost == 2.5
    assert rejected

def test_semantic_violation_cost_no_edges():
    """
    Test with a graph mapping containing no edges.
    """
    calc = SemanticViolationCostCalculator(threshold=2.0)
    graph = {"nodes": [{"id": "A"}]}
    cost, rejected = calc.evaluate(graph)
    assert cost == 0.0
    assert not rejected
