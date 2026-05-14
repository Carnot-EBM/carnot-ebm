"""SCG-MEM structural enforcer for Carnot's FR-11 self-learning traces."""
import json
from typing import Dict, Any, List

class ScgAdapter:
    """Enforces strict schema limits on continuous memory using SCG-MEM."""
    
    def __init__(self, schema: Dict[str, Any] = None):
        if schema is None:
            self.schema = {
                "type": "object",
                "properties": {
                    "trace_id": {"type": "string"},
                    "memory_embedding": {"type": "array", "items": {"type": "number"}},
                    "cognitive_context": {"type": "string"},
                    "utility_score": {"type": "number"}
                },
                "required": ["trace_id", "memory_embedding", "cognitive_context"]
            }
        else:
            self.schema = schema
            
    def enforce_schema(self, trace_data: Dict[str, Any]) -> bool:
        """
        Validates the trace data against the SCG-MEM cognitive schema.
        In a real implementation, this would use jsonschema.validate.
        Here we do a basic structural check based on the required fields and types.
        """
        if not isinstance(trace_data, dict):
            return False
            
        required_fields = self.schema.get("required", [])
        for field in required_fields:
            if field not in trace_data:
                return False
                
        # Basic type checking for properties
        properties = self.schema.get("properties", {})
        for key, expected_type_info in properties.items():
            if key in trace_data:
                val = trace_data[key]
                expected_type = expected_type_info.get("type")
                if expected_type == "string" and not isinstance(val, str):
                    return False
                if expected_type == "number" and not isinstance(val, (int, float)):
                    return False
                if expected_type == "array" and not isinstance(val, list):
                    return False
                # If array, check items
                if expected_type == "array" and "items" in expected_type_info:
                    item_type = expected_type_info["items"].get("type")
                    if item_type == "number" and not all(isinstance(x, (int, float)) for x in val):
                        return False
                        
        return True

    def process_embeddings(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes a list of traces and returns only those that strictly adhere to the schema."""
        valid_traces = []
        for trace in traces:
            if self.enforce_schema(trace):
                valid_traces.append(trace)
        return valid_traces
