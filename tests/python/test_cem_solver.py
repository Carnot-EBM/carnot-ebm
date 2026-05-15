"""Tests for CEM Solver.

Spec: REQ-CEM-004, REQ-CEM-005, SCENARIO-CEM-002
"""

from carnot.verify.plan_graph_energy_adapter import PlanGraph, PlanNode, PlanEdge
from carnot.cem.decomposition import decompose_plan_graph
from carnot.cem.solver import (
    minimize_landscape,
    parallel_minimize,
    compute_global_energy,
)

def test_minimize_landscape():
    """Test minimizing a single localized landscape."""
    node1 = PlanNode("n1", "prompt", "n1", "string", {})
    node2 = PlanNode("n2", "tool_call", "n2", "string", {})
    edge1 = PlanEdge("e1", "n1", "n2", "triggers", {})
    
    graph = PlanGraph("case1", "test_family", (node1, node2), (edge1,), {})
    landscapes = decompose_plan_graph(graph)
    
    assert len(landscapes) == 3
    
    minimized = minimize_landscape(landscapes[0])
    assert minimized.landscape_id == "case1:parsing"
    assert minimized.energy >= 0.0

def test_parallel_minimize():
    """Test parallel minimization of multiple landscapes."""
    node1 = PlanNode("n1", "prompt", "n1", "string", {})
    node2 = PlanNode("n2", "tool_call", "n2", "string", {})
    node3 = PlanNode("n3", "tool_result", "n3", "string", {})
    node4 = PlanNode("n4", "final_answer", "n4", "string", {})
    edge1 = PlanEdge("e1", "n1", "n2", "triggers", {})
    edge2 = PlanEdge("e2", "n2", "n3", "returns", {})
    edge3 = PlanEdge("e3", "n3", "n4", "triggers", {})
    
    graph = PlanGraph("case2", "test_family", (node1, node2, node3, node4), (edge1, edge2, edge3), {})
    landscapes = decompose_plan_graph(graph)
    
    minimized_list = parallel_minimize(landscapes)
    assert len(minimized_list) == len(landscapes)
    
    global_energy = compute_global_energy(minimized_list)
    assert global_energy >= 0.0
