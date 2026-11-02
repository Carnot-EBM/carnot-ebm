import re
import os
import json
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Any, Dict

@dataclass
class DynamicConstraint:
    instruction_type: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def check(self, response: str) -> bool:
        if self.instruction_type == "must_contain":
            return self.metadata.get("term", "") in response
        elif self.instruction_type == "must_not_contain":
            return self.metadata.get("term", "") not in response
        elif self.instruction_type == "format_json":
            try:
                json.loads(response)
                return True
            except json.JSONDecodeError:
                return False
        elif self.instruction_type == "format_list":
            # Match hyphen, asterisk, plus, or numbered lists at the start of lines
            return bool(re.search(r'^\s*[-*+]\s', response, re.MULTILINE)) or bool(re.search(r'^\s*\d+\.\s', response, re.MULTILINE))
        elif self.instruction_type == "max_words":
            words = len(response.split())
            return words <= self.metadata.get("max", float('inf'))
        elif self.instruction_type == "min_words":
            words = len(response.split())
            return words >= self.metadata.get("min", 0)
        elif self.instruction_type == "numeric_range":
            # Find numbers and check if they are in range
            numbers = [float(n) for n in re.findall(r'-?\d+\.?\d*', response)]
            min_val = self.metadata.get("min", float('-inf'))
            max_val = self.metadata.get("max", float('inf'))
            return all(min_val <= n <= max_val for n in numbers) if numbers else False
        elif self.instruction_type == "starts_with":
            return response.strip().startswith(self.metadata.get("term", ""))
        elif self.instruction_type == "ends_with":
            return response.strip().endswith(self.metadata.get("term", ""))
        elif self.instruction_type == "no_repetition":
            words = response.lower().split()
            return len(words) == len(set(words))
        return True

class PromptConstraintExtractor:
    def __init__(self, generate_fn: Optional[Callable] = None):
        self.generate_fn = generate_fn
        
    def extract_from_prompt(self, prompt: str) -> List[DynamicConstraint]:
        constraints = []
        
        # must_contain
        m = re.search(r"must include the word '(.+?)'", prompt, re.IGNORECASE)
        if m:
            constraints.append(DynamicConstraint("must_contain", "Response must contain specific word", {"term": m.group(1)}))
            
        # must_not_contain
        m = re.search(r"must not include the word '(.+?)'", prompt, re.IGNORECASE)
        if m:
            constraints.append(DynamicConstraint("must_not_contain", "Response must not contain specific word", {"term": m.group(1)}))
            
        # format_json
        if re.search(r"format as json", prompt, re.IGNORECASE):
            constraints.append(DynamicConstraint("format_json", "Response must be valid JSON"))
            
        # format_list
        if re.search(r"format as list", prompt, re.IGNORECASE):
            constraints.append(DynamicConstraint("format_list", "Response must be formatted as a list"))
            
        # max_words
        m = re.search(r"maximum of (\d+) words", prompt, re.IGNORECASE)
        if m:
            constraints.append(DynamicConstraint("max_words", f"Response must have maximum {m.group(1)} words", {"max": int(m.group(1))}))
            
        # min_words
        m = re.search(r"minimum of (\d+) words", prompt, re.IGNORECASE)
        if m:
            constraints.append(DynamicConstraint("min_words", f"Response must have minimum {m.group(1)} words", {"min": int(m.group(1))}))
            
        # numeric_range
        m = re.search(r"between (\d+) and (\d+)", prompt, re.IGNORECASE)
        if m:
            constraints.append(DynamicConstraint("numeric_range", f"Numbers must be between {m.group(1)} and {m.group(2)}", {"min": float(m.group(1)), "max": float(m.group(2))}))
            
        # starts_with
        m = re.search(r"starts with '(.+?)'", prompt, re.IGNORECASE)
        if m:
            constraints.append(DynamicConstraint("starts_with", f"Response must start with '{m.group(1)}'", {"term": m.group(1)}))
            
        # ends_with
        m = re.search(r"ends with '(.+?)'", prompt, re.IGNORECASE)
        if m:
            constraints.append(DynamicConstraint("ends_with", f"Response must end with '{m.group(1)}'", {"term": m.group(1)}))
            
        # no_repetition
        if re.search(r"no repetition", prompt, re.IGNORECASE):
            constraints.append(DynamicConstraint("no_repetition", "Response must not contain repeating words"))

        if os.environ.get("CARNOT_FORCE_LIVE") == "1" and self.generate_fn:
            llm_constraints = self.generate_fn(prompt)
            if llm_constraints:
                constraints.extend(llm_constraints)
                
        return constraints

    def check_response(self, response: str, constraints: List[DynamicConstraint]) -> List[DynamicConstraint]:
        violated = []
        for c in constraints:
            if not c.check(response):
                violated.append(c)
        return violated
