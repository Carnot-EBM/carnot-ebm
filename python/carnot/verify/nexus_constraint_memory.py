import json
import os
from collections import defaultdict
from typing import Dict, List, Any


class NexusConstraintMemory:
    """
    A memory module for capturing and synthesizing symbolic constraint rules 
    from empirical violation patterns.
    """

    def __init__(self) -> None:
        # Map domain -> pattern -> list of severities
        self.violations: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.rules: List[Dict[str, Any]] = []

    def record_violation(self, pattern: str, domain: str, severity: float) -> None:
        """
        Record a violation event.
        """
        self.violations[domain][pattern].append(severity)

    def synthesize_rules(self) -> List[Dict[str, Any]]:
        """
        Synthesize rules from patterns seen >= 3 times.
        Returns the synthesized rules.
        """
        self.rules = []
        for domain, patterns in self.violations.items():
            for pattern, severities in patterns.items():
                if len(severities) >= 3:
                    avg_severity = sum(severities) / len(severities)
                    self.rules.append({
                        "pattern": pattern,
                        "domain": domain,
                        "count": len(severities),
                        "avg_severity": avg_severity
                    })
        return self.rules

    def consolidate(self) -> None:
        """
        Merge redundant/identical rules.
        """
        consolidated = {}
        for rule in self.rules:
            key = (rule["domain"], rule["pattern"])
            if key not in consolidated:
                consolidated[key] = rule.copy()
            else:
                existing = consolidated[key]
                # Merge counts and average severity
                total_count = existing["count"] + rule["count"]
                total_sev = (existing["avg_severity"] * existing["count"] + rule["avg_severity"] * rule["count"])
                existing["avg_severity"] = total_sev / total_count
                existing["count"] = total_count

        self.rules = list(consolidated.values())

    def save(self, path: str) -> None:
        """
        Save memory state to a JSON file.
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {
            "violations": {d: dict(p) for d, p in self.violations.items()},
            "rules": self.rules
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """
        Load memory state from a JSON file.
        """
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        
        self.violations.clear()
        for d, p_dict in data.get("violations", {}).items():
            for p, sevs in p_dict.items():
                self.violations[d][p] = sevs
                
        self.rules = data.get("rules", [])
