from typing import Dict, List, Optional, Tuple

class TrieNode:
    def __init__(self, token_id: Optional[int], prob: float):
        self.token_id = token_id
        self.prob = prob
        self.children: Dict[int, 'TrieNode'] = {}
        self.is_terminal = False
        self.is_pruned = False

class PrefixClosedBoundVerifier:
    def __init__(self):
        self.root = TrieNode(None, 1.0)
        self.frontier: List[Tuple[Tuple[int, ...], float]] = [((), 1.0)]
        self.pruned_mass = 0.0
        self.terminal_success_mass = 0.0

    def _get_node(self, prefix: Tuple[int, ...]) -> Optional[TrieNode]:
        node = self.root
        for token in prefix:
            if token not in node.children:
                return None
            node = node.children[token]
        return node

    def add_expansion(self, prefix: Tuple[int, ...], expansions: Dict[int, float], is_violation: bool, is_terminal: bool = False):
        """
        Expand a node in the trie.
        If is_violation is True, the node deterministically violates constraints.
        """
        node = self._get_node(prefix)
        if node is None:
            # Create path if it doesn't exist
            node = self.root
            path_prob = 1.0
            for token in prefix:
                if token not in node.children:
                    node.children[token] = TrieNode(token, 1.0) # Dummy prob, should be set properly
                node = node.children[token]

        if node.is_pruned:
            return # Already pruned, do nothing

        if is_violation:
            node.is_pruned = True
            # Compute the path mass
            path_mass = 1.0
            curr = self.root
            for token in prefix:
                curr = curr.children[token]
                path_mass *= curr.prob
            self.pruned_mass += path_mass
            return

        if is_terminal:
            node.is_terminal = True
            # Compute path mass
            path_mass = 1.0
            curr = self.root
            for token in prefix:
                curr = curr.children[token]
                path_mass *= curr.prob
            self.terminal_success_mass += path_mass
            return
            
        # Add children
        for token, prob in expansions.items():
            if token not in node.children:
                child = TrieNode(token, prob)
                node.children[token] = child

    def compute_bounds(self) -> Tuple[float, float]:
        """
        Returns (lower_bound, upper_bound)
        lower_bound: mass of known terminal success paths.
        upper_bound: 1.0 - mass of known pruned paths.
        """
        lower_bound = round(self.terminal_success_mass, 10)
        upper_bound = round(1.0 - self.pruned_mass, 10)
        return lower_bound, upper_bound

    def check_monotonicity(self, previous_bounds: Tuple[float, float]) -> bool:
        """
        Validate that the bounds are monotonic.
        lower_bound should only increase, upper_bound should only decrease.
        """
        curr_lower, curr_upper = self.compute_bounds()
        prev_lower, prev_upper = previous_bounds
        
        # Using a small epsilon for floating point comparisons
        eps = 1e-9
        return (curr_lower >= prev_lower - eps) and (curr_upper <= prev_upper + eps)

    def sample_estimate(self, num_samples: int, evaluate_fn) -> Tuple[float, float]:
        """
        Compute a loose sampling bound.
        """
        successes = 0
        for _ in range(num_samples):
            if evaluate_fn():
                successes += 1
        
        estimate = successes / num_samples
        # Simple loose bound based on sampling (not a true confidence interval here for simplicity)
        return estimate, estimate
