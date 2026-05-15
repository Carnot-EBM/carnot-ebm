"""Tests for CEM decomposition engine."""

from carnot.cem.decomposition import decompose_plan_graph, LocalizedLandscape
from carnot.verify.plan_graph_energy_adapter import convert_cctu_traces_to_plan_graphs

def test_cem_decomposition_req_cem_001_and_002():
    """Verify REQ-CEM-001 and REQ-CEM-002: decomposes graphs on existing CCTU traces."""
    graphs = convert_cctu_traces_to_plan_graphs(trace_limit=2)
    assert len(graphs) > 0
    
    for graph in graphs:
        landscapes = decompose_plan_graph(graph)
        assert len(landscapes) == 3
        
        assert landscapes[0].landscape_id.endswith(":parsing")
        assert len(landscapes[0].nodes) > 0
        
        assert landscapes[1].landscape_id.endswith(":execution")
        assert len(landscapes[1].nodes) == 2
        
        assert landscapes[2].landscape_id.endswith(":formatting")
        assert len(landscapes[2].nodes) == 2
