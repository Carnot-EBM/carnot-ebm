"""LLM-assisted arithmetic extraction for free-form reasoning traces.

Prompts a small auxiliary model to rewrite natural-language arithmetic into
canonical claim lines, then verifies each extracted claim deterministically and
adapts it to Carnot's `ConstraintResult` / `ConstraintTerm` pipeline.

Spec: REQ-VERIFY-001, REQ-VERIFY-003, REQ-VERIFY-010, SCENARIO-VERIFY-010
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Callable

import jax.numpy as jnp

from carnot.pipeline.extract import ConstraintResult
from carnot.verify.constraint import BaseConstraint

LoadModelFn = Callable[[str], tuple[Any, Any]]
GenerateFn = Callable[[Any, Any, str, int], str]

_NUMBER = r"-?\d+(?:,\d{3})*(?:\.\d+)?"
_CLAIM_PATTERN = re.compile(
    rf"CLAIM:\s*(?P<a>{_NUMBER})\s*(?P<op>[+\-*/])\s*(?P<b>{_NUMBER})\s*=\s*(?P<c>{_NUMBER})",
    re.IGNORECASE,
)


def _default_load_model(model_name: str) -> tuple[Any, Any]:
    from carnot.inference.model_loader import load_model

    return load_model(model_name)


def _default_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> str:
    from carnot.inference.model_loader import generate

    return generate(model, tokenizer, prompt, max_new_tokens)


class _LLMArithmeticClaimConstraint(BaseConstraint):
    """Fixed-energy adapter for one LLM-extracted arithmetic claim."""

    def __init__(self, claim_satisfied: bool, claim_description: str) -> None:
        self._satisfied = claim_satisfied
        self._description = claim_description

    @property
    def name(self) -> str:
        status = "ok" if self._satisfied else "violated"
        return f"llm_claim({status}): {self._description[:50]}"

    @property
    def satisfaction_threshold(self) -> float:
        return 0.5

    def energy(self, x: Any) -> Any:
        _ = x
        return jnp.float32(0.0 if self._satisfied else 1.0)

    def is_satisfied(self, x: Any) -> bool:
        _ = x
        return self._satisfied


class LLMConstraintExtractor:
    """Use an auxiliary LLM call to canonicalize arithmetic claims."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        model: Any = None,
        tokenizer: Any = None,
        load_model_fn: LoadModelFn | None = None,
        generate_fn: GenerateFn | None = None,
        max_new_tokens: int = 128,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._model_name = model_name
        self._model = model
        self._tokenizer = tokenizer
        self._load_model_fn = load_model_fn or _default_load_model
        self._generate_fn = generate_fn or _default_generate
        self._max_new_tokens = max_new_tokens
        self._clock = clock
        self.last_latency_seconds = 0.0

    @property
    def supported_domains(self) -> list[str]:
        return ["arithmetic"]

    def extract(
        self, text: str, domain: str | None = None
    ) -> list[ConstraintResult]:
        if domain is not None and domain not in self.supported_domains:
            self.last_latency_seconds = 0.0
            return []
        if not text.strip():
            self.last_latency_seconds = 0.0
            return []

        model, tokenizer = self._ensure_model()
        prompt = self._build_prompt(text)
        start = self._clock()
        raw_output = self._generate_fn(
            model,
            tokenizer,
            prompt,
            self._max_new_tokens,
        )
        self.last_latency_seconds = self._clock() - start
        return self._parse_claims(raw_output, self.last_latency_seconds)

    def _ensure_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        if self._model_name is None:
            raise RuntimeError(
                "LLMConstraintExtractor requires model_name or preloaded "
                "model/tokenizer."
            )

        model, tokenizer = self._load_model_fn(self._model_name)
        if model is None or tokenizer is None:
            raise RuntimeError(
                f"Could not load extractor model '{self._model_name}'."
            )

        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer

    @staticmethod
    def _build_prompt(text: str) -> str:
        return (
            "You extract verifiable arithmetic claims from model responses.\n"
            "List all verifiable arithmetic claims in this response.\n"
            "Format each as: CLAIM: a OP b = c\n"
            "Use one claim per line.\n"
            "Only output CLAIM lines with numeric operands/results and "
            "operators +, -, *, /.\n"
            "If there are no verifiable arithmetic claims, output NONE.\n\n"
            f"Response:\n{text}"
        )

    def _parse_claims(
        self,
        raw_output: str,
        latency_seconds: float,
    ) -> list[ConstraintResult]:
        results: list[ConstraintResult] = []
        for line in raw_output.splitlines():
            match = _CLAIM_PATTERN.search(line)
            if match is None:
                continue

            a = self._parse_number(match.group("a"))
            b = self._parse_number(match.group("b"))
            claimed = self._parse_number(match.group("c"))
            operator = match.group("op")

            try:
                correct = self._compute_result(a, operator, b)
            except ZeroDivisionError:
                continue

            satisfied = self._results_match(claimed, correct)
            raw_claim = (
                f"{self._format_number(a)} {operator} "
                f"{self._format_number(b)} = {self._format_number(claimed)}"
            )
            description = raw_claim
            if not satisfied:
                description += f" (correct: {self._format_number(correct)})"

            results.append(
                ConstraintResult(
                    constraint_type="arithmetic",
                    description=description,
                    energy_term=_LLMArithmeticClaimConstraint(
                        claim_satisfied=satisfied,
                        claim_description=raw_claim,
                    ),
                    metadata={
                        "a": a,
                        "b": b,
                        "operator": operator,
                        "claimed_result": claimed,
                        "correct_result": correct,
                        "satisfied": satisfied,
                        "raw_claim": raw_claim,
                        "extractor": "llm",
                        "extraction_latency_seconds": latency_seconds,
                    },
                )
            )
        return results

    @staticmethod
    def _parse_number(text: str) -> int | float:
        cleaned = text.replace(",", "")
        if "." in cleaned:
            return float(cleaned)
        return int(cleaned)

    @staticmethod
    def _compute_result(a: int | float, operator: str, b: int | float) -> int | float:
        if operator == "+":
            result = a + b
        elif operator == "-":
            result = a - b
        elif operator == "*":
            result = a * b
        else:
            result = a / b

        if isinstance(result, float) and result.is_integer():
            return int(result)
        return result

    @staticmethod
    def _results_match(claimed: int | float, correct: int | float) -> bool:
        if isinstance(claimed, int) and isinstance(correct, int):
            return claimed == correct
        return math.isclose(
            float(claimed),
            float(correct),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

    @staticmethod
    def _format_number(value: int | float) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
