"""Constraint-Aware Retrieval Module (CARM).

References: ConstraintLLM paper, Milestone 2026.05.211.
"""

from __future__ import annotations

import re


class CARM:
    """Constraint-Aware Retrieval Mechanism for formal constraints.
    
    Dynamically retrieves verifiable logic schemas (domains and constraint
    types) based on semantic matching of natural language prompt tokens
    against the existing static constraint library.
    """

    DOMAIN_KEYWORDS = {
        "arithmetic": ["add", "subtract", "sum", "math", "calculate", "equation", "+", "-", "="],
        "code": ["python", "code", "function", "variable", "return", "loop", "type", "def"],
        "logic": ["if", "then", "must", "cannot", "either", "or", "all", "not", "exclude", "imply", "implies"],
        "nl": ["is", "are", "has", "have", "there", "quantity", "factual"],
    }

    CONSTRAINT_TYPE_KEYWORDS = {
        "arithmetic": ["add", "sum", "subtract", "math", "+", "-"],
        "type_check": ["type", "annotation", "parameter type"],
        "return_type": ["returns", "return type", "output"],
        "bound": ["bound", "limit", "range", "between", "0 <="],
        "initialization": ["initialize", "assign", "defined"],
        "implication": ["if", "then", "implies"],
        "exclusion": ["but not", "exclude", "without", "cannot"],
        "disjunction": ["either", "or"],
        "negation": ["cannot", "not", "never", "do not", "does not"],
        "universal": ["all", "every", "always"],
        "factual": ["is", "are"],
        "factual_relation": ["is the", "relation"],
        "quantity": ["how many", "count", "number of", "there are"],
    }

    def __init__(self) -> None:
        """Initialize CARM."""
        pass

    def retrieve_domains(self, prompt: str) -> list[str]:
        """Dynamically retrieve verifiable logic domains based on prompt tokens."""
        prompt_lower = prompt.lower()
        tokens = set(re.findall(r"\b\w+\b", prompt_lower))
        
        matched_domains = set()
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if kw in tokens or kw in prompt_lower:
                    matched_domains.add(domain)
                    break
                    
        return sorted(list(matched_domains))
        
    def retrieve_constraint_types(self, prompt: str) -> list[str]:
        """Dynamically retrieve verifiable constraint types based on prompt tokens."""
        prompt_lower = prompt.lower()
        tokens = set(re.findall(r"\b\w+\b", prompt_lower))
        
        matched_types = set()
        for ctype, keywords in self.CONSTRAINT_TYPE_KEYWORDS.items():
            for kw in keywords:
                if kw in prompt_lower or kw in tokens:
                    matched_types.add(ctype)
                    break
                    
        return sorted(list(matched_types))
