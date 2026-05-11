"""
ROCE Open Constraint Elicitation Prototype.

Spec: REQ-ROCE-1864, SCENARIO-ROCE-1864
"""
import re
import json
from typing import Any

class GenerationLogicExtractor:
    """Extracts structured logic from unconstrained SOTA generation output."""
    def __init__(self) -> None:
        self.model_target = "unsloth/Qwen3.6-35B-A3B-GGUF"

    def extract_logic(self, generation_text: str) -> dict[str, Any]:
        """Extract verifiable logic dynamically from the model's natural language output."""
        constraints = []
        
        json_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', generation_text, re.DOTALL)
        for block in json_blocks:
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict) and "constraints" in parsed:
                    constraints.extend(parsed["constraints"])
                else:
                    constraints.append(parsed)
            except json.JSONDecodeError:
                pass
                
        if not constraints:
            match = re.search(r'(?i)must\s+contain\s+["\']?([^"\'.]+)["\']?', generation_text)
            if match:
                constraints.append({"type": "contains", "value": match.group(1)})
                
        return {"model": self.model_target, "extracted_constraints": constraints, "success": len(constraints) > 0}
        
    def evaluate(self, dataset: list[str]) -> dict[str, Any]:
        """Evaluate extraction success rate on a dataset."""
        results = []
        successes = 0
        for text in dataset:
            result = self.extract_logic(text)
            results.append({"input": text, "output": result})
            if result["success"]:
                successes += 1
                
        success_rate = successes / len(dataset) if dataset else 0.0
        return {
            "dataset_size": len(dataset),
            "successes": successes,
            "success_rate": success_rate,
            "results": results
        }
