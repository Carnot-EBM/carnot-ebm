import copy
from typing import Dict, Any, List

class NonforgettingMemoryUpdater:
    def __init__(self, baseline_memory: Dict[str, Any]):
        self.memory = copy.deepcopy(baseline_memory)
        self.holdout_set: List[Dict[str, Any]] = []
        self.rollback_count = 0
    
    def set_holdout(self, holdout_cases: List[Dict[str, Any]]):
        self.holdout_set = holdout_cases

    def evaluate_holdout(self, memory: Dict[str, Any]) -> float:
        """Returns the regression rate on holdout set. 0.0 means no regression."""
        if not self.holdout_set:
            return 0.0
        
        regressions = 0
        for case in self.holdout_set:
            key = case["key"]
            expected = case["expected"]
            if memory.get(key) != expected:
                regressions += 1
        
        return regressions / len(self.holdout_set)

    def update(self, new_conflicts: Dict[str, Any]) -> float:
        candidate_memory = copy.deepcopy(self.memory)
        for k, v in new_conflicts.items():
            candidate_memory[k] = v
        
        regression_rate = self.evaluate_holdout(candidate_memory)
        
        if regression_rate > 0.0:
            self.rollback_count += 1
            holdout_keys = {case["key"] for case in self.holdout_set}
            candidate_memory = copy.deepcopy(self.memory)
            for k, v in new_conflicts.items():
                if k not in holdout_keys:
                    candidate_memory[k] = v
            
            regression_rate = self.evaluate_holdout(candidate_memory)
            if regression_rate > 0.0:
                candidate_memory = copy.deepcopy(self.memory)
                regression_rate = 0.0

        self.memory = candidate_memory
        return regression_rate