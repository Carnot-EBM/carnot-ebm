import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class VerificationLearningProxy:
    """
    Verification Learning (VL) proxy for continuous self-learning.
    
    Implements a constraint-based loss function that operates natively on unlabelled data 
    without labeled targets, based on arXiv:2503.12917.
    """
    def __init__(self, constraints: Optional[List[Dict[str, Any]]] = None):
        self.constraints = constraints or []
        
    def score_constraint_satisfaction(self, unlabelled_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Scores unlabelled data directly based on constraint satisfaction.
        Returns a dictionary of data identifiers to their constraint proxy scores.
        """
        scores = {}
        for item in unlabelled_data:
            item_id = item.get("id", "unknown")
            text = item.get("text", "")
            
            # Simple native proxy: if there are no constraints, score is 1.0 (perfect)
            if not self.constraints:
                scores[item_id] = 1.0
                continue
                
            # Otherwise, count how many constraints are satisfied.
            # A constraint might define a "must_contain" or "must_not_contain" rule.
            satisfied_count = 0
            for constraint in self.constraints:
                c_type = constraint.get("type")
                c_value = constraint.get("value", "")
                
                if c_type == "must_contain":
                    if c_value in text:
                        satisfied_count += 1
                elif c_type == "must_not_contain":
                    if c_value not in text:
                        satisfied_count += 1
                else:
                    # Unknown constraint type, assume satisfied for the proxy
                    satisfied_count += 1
            
            scores[item_id] = float(satisfied_count) / len(self.constraints)
            
        return scores

    def compute_proxy_loss(self, unlabelled_data: List[Dict[str, Any]]) -> float:
        """
        Computes the verification learning proxy loss natively.
        Loss is calculated as (1.0 - average_satisfaction_score).
        """
        if not unlabelled_data:
            return 0.0
            
        scores = self.score_constraint_satisfaction(unlabelled_data)
        average_score = sum(scores.values()) / len(scores)
        return 1.0 - average_score

    def run_experiment_and_save(self, unlabelled_data: List[Dict[str, Any]], result_path: str | Path) -> Dict[str, Any]:
        """
        Runs the proxy evaluation on unlabelled data and writes results to JSON.
        Traces to REQ-LEARN-1854.
        """
        path = Path(result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        scores = self.score_constraint_satisfaction(unlabelled_data)
        loss = self.compute_proxy_loss(unlabelled_data)
        
        # Determine honest_verdict
        honest_verdict = "vl_proxy_success" if loss < 0.5 else "vl_proxy_needs_improvement"
        if not unlabelled_data:
             honest_verdict = "vl_proxy_empty_data"
             
        result = {
            "experiment_id": "1854",
            "honest_verdict": honest_verdict,
            "proxy_loss": loss,
            "scores": scores,
            "constraint_count": len(self.constraints),
            "data_count": len(unlabelled_data)
        }
        
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
            
        return result
