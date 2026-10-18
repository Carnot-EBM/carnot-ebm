"""CEM Logic Decomposition Engine.

Spec: REQ-CEM-001, REQ-CEM-002
"""

from dataclasses import dataclass

from carnot.verify.plan_graph_energy_adapter import PlanGraph, PlanNode, PlanEdge


@dataclass(frozen=True)
class LocalizedLandscape:
    """A localized energy landscape decomposed from a monolithic constraint graph."""
    landscape_id: str
    nodes: tuple[PlanNode, ...]
    edges: tuple[PlanEdge, ...]


def decompose_plan_graph(graph: PlanGraph) -> list[LocalizedLandscape]:
    """Decompose a PlanGraph into localized energy landscapes.
    
    This implements REQ-CEM-001 by splitting the monolithic constraint graph
    into smaller subsets, grouped by step logic (e.g., prompt parsing, execution, formatting).
    """
    landscapes = []
    
    # Group 1: prompt -> tool_call
    prompt_nodes = tuple(
        n for n in graph.nodes if n.node_type in ("prompt", "tool_argument", "tool_call")
    )
    prompt_node_ids = {n.node_id for n in prompt_nodes}
    prompt_edges = tuple(
        e for e in graph.edges if e.source in prompt_node_ids and e.target in prompt_node_ids
    )
    landscapes.append(LocalizedLandscape(f"{graph.case_id}:parsing", prompt_nodes, prompt_edges))
    
    # Group 2: tool_call -> tool_result
    exec_nodes = tuple(
        n for n in graph.nodes if n.node_type in ("tool_call", "tool_result")
    )
    exec_node_ids = {n.node_id for n in exec_nodes}
    exec_edges = tuple(
        e for e in graph.edges if e.source in exec_node_ids and e.target in exec_node_ids
    )
    landscapes.append(LocalizedLandscape(f"{graph.case_id}:execution", exec_nodes, exec_edges))
    
    # Group 3: tool_result -> final_answer
    fmt_nodes = tuple(
        n for n in graph.nodes if n.node_type in ("tool_result", "final_answer")
    )
    fmt_node_ids = {n.node_id for n in fmt_nodes}
    fmt_edges = tuple(
        e for e in graph.edges if e.source in fmt_node_ids and e.target in fmt_node_ids
    )
    landscapes.append(LocalizedLandscape(f"{graph.case_id}:formatting", fmt_nodes, fmt_edges))
    
    return landscapes
