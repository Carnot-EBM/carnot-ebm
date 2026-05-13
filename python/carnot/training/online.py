"""Online training promotion loop for FR-11 with no-forgetting constraints.

Spec: REQ-LEARN-1986
"""

from typing import Dict, Any, List

class ValidatorTreeLedger:
    """Ledger for validator-tree promotions ensuring routing without forgetting."""
    
    def __init__(self) -> None:
        self.soundness_mistakes: int = 0
        self.completeness_mistakes: int = 0
        self.history: List[Dict[str, Any]] = []

    def evaluate_promotion(
        self,
        utility_delta: float,
        soundness_mistakes: int,
        completeness_mistakes: int,
        previous_performance: float,
        new_performance: float
    ) -> Dict[str, Any]:
        """
        Evaluate a model promotion against FR-11 non-forgetting constraints.
        
        A promotion is only passed if:
        1. utility_delta > 0
        2. Non-forgetting holds (new_performance >= previous_performance)
        """
        self.soundness_mistakes += soundness_mistakes
        self.completeness_mistakes += completeness_mistakes

        non_forgetting_holds = new_performance >= previous_performance
        
        gate_passed = False
        if utility_delta > 0 and non_forgetting_holds:
            gate_passed = True

        result = {
            "promotion_gate_passed": gate_passed,
            "utility_delta": utility_delta,
            "soundness_mistakes": self.soundness_mistakes,
            "completeness_mistakes": self.completeness_mistakes,
            "non_forgetting_holds": non_forgetting_holds,
            "previous_performance": previous_performance,
            "new_performance": new_performance
        }
        
        self.history.append(result)
        return result
