from typing import Dict, Any, Tuple

class SemanticViolationCostCalculator:
    """
    Calculates the structural deformation cost for reasoning outputs mapped
    to a knowledge graph to detect and reject smooth falsehoods.
    """
    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold

    def calculate_cost(self, graph_mapping: Dict[str, Any]) -> float:
        """
        Calculate structural deformation cost based on edge violations.
        """
        cost = 0.0
        edges = graph_mapping.get('edges', [])
        for edge in edges:
            if 'violation_weight' in edge:
                cost += edge['violation_weight']
            elif edge.get('status') == 'contradiction':
                cost += 1.0
        return cost

    def evaluate(self, graph_mapping: Dict[str, Any]) -> Tuple[float, bool]:
        """
        Evaluate the graph mapping and return (cost, rejected).
        The output is rejected if the semantic violation cost exceeds the threshold.
        """
        cost = self.calculate_cost(graph_mapping)
        return cost, cost > self.threshold
