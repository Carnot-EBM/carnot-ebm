import os
import logging
from typing import Callable, Dict, Any

logger = logging.getLogger(__name__)

COMPILER_PROMPT = """\
You are an expert Python compiler for natural language constraints.
Translate the following natural language rules into an executable Python function named `validate_constraints(assignment: dict) -> bool`.
The `assignment` argument is a dictionary mapping variable names (strings) to boolean values.
The function should return True if the assignment satisfies the rules, and False otherwise.
Return ONLY the raw Python code. Do not include markdown code blocks.

Rules:
{rules}
"""

class ConstrainPromptCompiler:
    def __init__(self, api_base: str = None, api_key: str = "not-needed", model: str = "unsloth/Qwen3.6-35B-A3B-GGUF"):
        self.api_base = api_base or os.environ.get("CARNOT_API_BASE", "http://localhost:8080/v1")
        self.api_key = api_key
        self.model = model

    def compile(self, rules: str) -> Callable[[Dict[str, Any]], bool]:
        """
        Compile natural language rules into an executable Python function.
        Spec: REQ-CONSTRAIN-001
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed")
            
        prompt = COMPILER_PROMPT.format(rules=rules)
        client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        
        raw_code = response.choices[0].message.content or ""
        
        if raw_code.startswith("```python"):
            raw_code = raw_code.split("```python")[1].split("```")[0].strip()
        elif raw_code.startswith("```"):
            raw_code = raw_code.split("```")[1].split("```")[0].strip()
            
        namespace = {}
        try:
            exec(raw_code, namespace)
            if "validate_constraints" not in namespace:
                raise ValueError("Generated code does not contain 'validate_constraints' function.")
            return namespace["validate_constraints"]
        except Exception as e:
            logger.error(f"Failed to compile rules. Error: {e}\nCode:\n{raw_code}")
            raise
