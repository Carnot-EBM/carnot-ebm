from typing import List, Set
import re


class SemanticPruner:
    """Semantic pruning mechanism for structural constraint rules.

    Identifies and removes redundant structural constraint rules based on
    semantic similarity (Jaccard similarity of normalized words) to reduce
    semantic redundancy in the FR-11 replay buffer.

    Spec: REQ-FR11-041
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def _normalize(self, text: str) -> Set[str]:
        """Normalize text and return a set of words."""
        # Convert to lowercase and extract alphanumeric words
        words = re.findall(r"\b\w+\b", text.lower())
        return set(words)

    def _calculate_similarity(self, rule1: str, rule2: str) -> float:
        """Calculate Jaccard similarity between two constraint rules.

        Spec: REQ-FR11-041-1
        """
        set1 = self._normalize(rule1)
        set2 = self._normalize(rule2)

        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)

    def prune_redundant_rules(self, rules: List[str]) -> List[str]:
        """Prune redundant rules from a list of constraint rules.

        Spec: REQ-FR11-041-2
        """
        if not rules:
            return []

        pruned_rules: List[str] = []
        for new_rule in rules:
            is_redundant = False
            for existing_rule in pruned_rules:
                similarity = self._calculate_similarity(new_rule, existing_rule)
                if similarity >= self.similarity_threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                pruned_rules.append(new_rule)

        return pruned_rules
