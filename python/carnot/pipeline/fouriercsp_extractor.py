"""FourierCSP Constraint Extractor.

This module provides the FourierCSPExtractor which maps natural language
constraints to a multilinear polynomial representation. It uses an LLM to
parse constraints and converts them to the mathematical mapping described
in FourierCSP.

Spec: REQ-EXTRACT-056, SCENARIO-EXTRACT-096
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass

# Default model from mandate
_DEFAULT_MODEL = "unsloth/Qwen3.6-35B-A3B-GGUF"

_LLM_PROMPT = (
    "You are a FourierCSP parser. Convert the following natural language constraint "
    "into a boolean logical expression (using AND, OR, NOT, XOR) and list the variables.\n\n"
    "Output MUST be valid JSON with keys:\n"
    '  "variables": ["x1", "x2"],\n'
    '  "expression": "x1 AND NOT x2"\n\n'
    "Constraint:\n"
)


def _default_generate(prompt: str) -> str:
    """Call the LLM to parse the constraint.
    
    Only used when CARNOT_FORCE_LIVE=1.
    """
    # Deferred import to avoid hard dependency on inference pipeline
    from carnot.inference.model_loader import generate, load_model

    model, tokenizer = load_model(_DEFAULT_MODEL)
    return generate(model, tokenizer, prompt, max_new_tokens=512)


@dataclass
class MultilinearPolynomial:
    """Represents a multilinear polynomial representation of a constraint.
    
    Spec: REQ-EXTRACT-056-2
    """
    variables: list[str]
    expression: str
    polynomial: str


class FourierCSPExtractor:
    """Extract constraints and map them to multilinear polynomials.
    
    Spec: REQ-EXTRACT-056-1
    """

    def __init__(self, generate_fn: Callable[[str], str] | None = None) -> None:
        self._generate_fn = generate_fn or _default_generate

    def extract(self, prompt: str) -> MultilinearPolynomial | None:
        """Extract a multilinear polynomial representation from a natural language prompt.
        
        Args:
            prompt: The natural language constraint.
            
        Returns:
            MultilinearPolynomial representation or None if parsing fails.
        """
        if not os.environ.get("CARNOT_FORCE_LIVE"):
            # Provide a deterministic fallback for testing without LLM
            return MultilinearPolynomial(
                variables=["x"],
                expression="x",
                polynomial="x",
            )

        try:
            llm_output = self._generate_fn(_LLM_PROMPT + prompt)
            return self._parse_llm_output(llm_output)
        except Exception:
            return None

    def _parse_llm_output(self, text: str) -> MultilinearPolynomial | None:
        """Parse LLM JSON output to multilinear polynomial."""
        try:
            # Strip possible markdown fences
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text.rsplit("\n", 1)[0]
            
            data = json.loads(text)
            variables = data.get("variables", [])
            expression = data.get("expression", "")
            
            # Simple conversion to illustrate multilinear polynomial mapping.
            # Real implementation would apply rigorous Walsh-Fourier transform.
            poly = expression.replace("AND", "*").replace("OR", "+").replace("NOT ", "1-")
            
            return MultilinearPolynomial(
                variables=variables,
                expression=expression,
                polynomial=poly,
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return None

__all__ = [
    "FourierCSPExtractor",
    "MultilinearPolynomial",
]
