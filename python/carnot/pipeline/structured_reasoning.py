"""Structured reasoning emission helpers for monitorable typed verification.

Spec: REQ-VERIFY-022, REQ-VERIFY-023, REQ-VERIFY-024,
SCENARIO-VERIFY-022, SCENARIO-VERIFY-023, SCENARIO-VERIFY-024
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from carnot.inference.model_loader import generate
from carnot.pipeline.typed_reasoning import TypedReasoningExtractor, TypedReasoningIR
from carnot.pipeline.typed_reasoning import extract_typed_reasoning as build_typed_reasoning_ir

RUN_DATE = "20260412"
_REQUIRED_KEYS = ("constraints", "steps", "claims", "final_answer")
_MODEL_ALIASES = {
    "qwen/qwen3.5-0.8b": "qwen",
    "qwen3.5-0.8b": "qwen",
    "google/gemma-4-e4b-it": "gemma",
    "gemma4-e4b-it": "gemma",
}
_SCHEMA_TEMPLATE = {
    "constraints": [
        {
            "constraint_id": "uc1",
            "kind": "prompt_constraint",
            "text": "<prompt constraint>",
        }
    ],
    "steps": [
        {
            "step_id": "s1",
            "kind": "analysis",
            "text": "<useful intermediate state>",
            "claim_ids": ["cl1"],
        }
    ],
    "claims": [
        {
            "claim_id": "cl1",
            "kind": "derived_fact",
            "text": "<short checkable claim>",
            "value": "<scalar, list, object, or null>",
            "step_id": "s1",
        }
    ],
    "final_answer": {
        "text": "<final answer>",
        "normalized": "<normalized value or null>",
        "answer_type": "text",
    },
}
_SCHEMA_SNIPPET = json.dumps(_SCHEMA_TEMPLATE, indent=2)


def get_repo_root() -> Path:
    """Resolve the repository root with the usual override for tests."""
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[3]


def default_policy_path() -> Path:
    """Return the Exp 213 monitorability policy path."""
    return get_repo_root() / "results" / "monitorability_policy_213.json"


def load_monitorability_policy(path: Path | None = None) -> dict[str, object]:
    """Load the Exp 213 policy, returning an empty dict when it is unavailable."""
    resolved = path or default_policy_path()
    if not resolved.exists():
        return {}

    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_qwen_structured_reasoning_prompt(question: str) -> str:
    """Build the minimal structured reasoning prompt for Qwen3.5-0.8B."""
    return (
        "Solve the task and return strict JSON only.\n"
        "Use this schema exactly:\n"
        f"{_SCHEMA_SNIPPET}\n"
        "Keep steps and claims brief. Include only verifier-visible constraints "
        "and useful intermediate facts.\n"
        "Use [] for empty lists. Do not add markdown or prose outside the JSON.\n\n"
        f"Task:\n{question}\n"
    )


def build_gemma_structured_reasoning_prompt(question: str) -> str:
    """Build the minimal structured reasoning prompt for Gemma4-E4B-it."""
    return (
        "Return a single JSON object that follows this schema:\n"
        f"{_SCHEMA_SNIPPET}\n"
        "Keep reasoning compact. Include only short, externally checkable "
        "steps and claims.\n"
        "Use [] for empty lists and null when a normalized value is unknown. "
        "Do not add markdown or extra text.\n\n"
        f"Task:\n{question}\n"
    )


def _normalize_model_name(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    return _MODEL_ALIASES.get(model_name.strip().lower())


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_json_payload(text: str) -> dict[str, object] | None:
    stripped = _strip_fence(text)
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


@dataclass(frozen=True)
class StructuredReasoningAttempt:
    """One structured emission attempt and its validation result."""

    prompt: str
    raw_response: str
    valid: bool
    error: str | None = None


@dataclass(frozen=True)
class StructuredReasoningEmission:
    """Final outcome of the structured reasoning controller."""

    policy_mode: str | None
    response_mode: str
    response: str
    typed_reasoning: TypedReasoningIR | None
    attempts: list[StructuredReasoningAttempt] = field(default_factory=list)
    fallback_used: bool = False


class StructuredReasoningController:
    """Policy-gated structured reasoning emitter with retry and fallback."""

    def __init__(
        self,
        policy: dict[str, object] | None = None,
        policy_path: Path | None = None,
        parser_version: str = RUN_DATE,
    ) -> None:
        self._policy = policy if policy is not None else load_monitorability_policy(policy_path)
        self._extractor = TypedReasoningExtractor(parser_version=parser_version)

    def recommended_mode(self, task_slice: str) -> str | None:
        """Return the Exp 213 recommended mode for a task slice when available."""
        per_task_slice = self._policy.get("per_task_slice")
        if not isinstance(per_task_slice, dict):
            return None
        entry = per_task_slice.get(task_slice)
        if not isinstance(entry, dict):
            return None
        mode = entry.get("recommended_mode")
        return mode if isinstance(mode, str) else None

    def should_use_structured_reasoning(self, task_slice: str, model_name: str | None) -> bool:
        """Only use structured prompting when policy and model support both allow it."""
        return (
            self.recommended_mode(task_slice) == "structured_json"
            and _normalize_model_name(model_name) is not None
        )

    def build_prompt(self, question: str, model_name: str | None) -> str:
        """Build the model-specific structured reasoning prompt."""
        normalized = _normalize_model_name(model_name)
        if normalized == "qwen":
            return build_qwen_structured_reasoning_prompt(question)
        if normalized == "gemma":
            return build_gemma_structured_reasoning_prompt(question)
        raise ValueError(f"unsupported model for structured reasoning prompt: {model_name}")

    def build_retry_prompt(self, question: str, model_name: str | None, error: str) -> str:
        """Request the same schema again while surfacing the last validation error."""
        return (
            f"Issue: {error}\n"
            "The previous response did not satisfy Carnot's structured reasoning schema.\n\n"
            f"{self.build_prompt(question, model_name)}"
        )

    def validate_response(self, question: str, response: str) -> TypedReasoningIR:
        """Validate a structured response before it is trusted by later verifiers."""
        payload = _extract_json_payload(response)
        if payload is None:
            raise ValueError("structured reasoning payload is not valid JSON")

        for key in _REQUIRED_KEYS:
            if key not in payload:
                raise ValueError(f"missing required top-level key: {key}")

        if not isinstance(payload["constraints"], list):
            raise ValueError("constraints must be a list")
        if not isinstance(payload["steps"], list):
            raise ValueError("steps must be a list")
        if not isinstance(payload["claims"], list):
            raise ValueError("claims must be a list")
        if not isinstance(payload["final_answer"], dict):
            raise ValueError("final_answer must be an object")

        ir = self._extractor.extract(question=question, response=response)
        if ir.provenance.extraction_method != "direct_json":
            raise ValueError("structured reasoning payload did not produce direct_json provenance")
        if ir.final_answer is None:
            raise ValueError("structured reasoning payload did not produce a final_answer")
        return ir

    def emit(
        self,
        question: str,
        task_slice: str,
        model_name: str | None,
        model: Any,
        tokenizer: Any,
        fallback_generate: Callable[[str, int], str] | None = None,
        max_attempts: int = 2,
        max_new_tokens: int = 220,
    ) -> StructuredReasoningEmission:
        """Emit monitorable structured reasoning when the Exp 213 policy recommends it."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        policy_mode = self.recommended_mode(task_slice)
        if not self.should_use_structured_reasoning(task_slice, model_name):
            return self._fallback(
                question=question,
                task_slice=task_slice,
                policy_mode=policy_mode,
                attempts=[],
                fallback_generate=fallback_generate,
                max_new_tokens=max_new_tokens,
            )

        prompt = self.build_prompt(question, model_name)
        attempts: list[StructuredReasoningAttempt] = []
        for _ in range(max_attempts):
            raw_response = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            try:
                ir = self.validate_response(question=question, response=raw_response)
            except ValueError as exc:
                attempts.append(
                    StructuredReasoningAttempt(
                        prompt=prompt,
                        raw_response=raw_response,
                        valid=False,
                        error=str(exc),
                    )
                )
                prompt = self.build_retry_prompt(question, model_name, str(exc))
                continue

            attempts.append(
                StructuredReasoningAttempt(
                    prompt=prompt,
                    raw_response=raw_response,
                    valid=True,
                )
            )
            return StructuredReasoningEmission(
                policy_mode=policy_mode,
                response_mode="structured_json",
                response=raw_response,
                typed_reasoning=ir,
                attempts=attempts,
                fallback_used=False,
            )

        return self._fallback(
            question=question,
            task_slice=task_slice,
            policy_mode=policy_mode,
            attempts=attempts,
            fallback_generate=fallback_generate,
            max_new_tokens=max_new_tokens,
        )

    def _fallback(
        self,
        question: str,
        task_slice: str,
        policy_mode: str | None,
        attempts: list[StructuredReasoningAttempt],
        fallback_generate: Callable[[str, int], str] | None,
        max_new_tokens: int,
    ) -> StructuredReasoningEmission:
        """Degrade to the caller's existing generation path without crashing."""
        del task_slice  # future task-slice-specific fallback prompts can use this

        if fallback_generate is None:
            return StructuredReasoningEmission(
                policy_mode=policy_mode,
                response_mode="unavailable",
                response="",
                typed_reasoning=None,
                attempts=list(attempts),
                fallback_used=True,
            )

        raw_response = fallback_generate(question, max_new_tokens)
        try:
            typed_reasoning = build_typed_reasoning_ir(question=question, response=raw_response)
        except Exception:
            typed_reasoning = None

        response_mode = (
            typed_reasoning.provenance.extraction_method
            if typed_reasoning is not None
            else "fallback_text"
        )
        return StructuredReasoningEmission(
            policy_mode=policy_mode,
            response_mode=response_mode,
            response=raw_response,
            typed_reasoning=typed_reasoning,
            attempts=list(attempts),
            fallback_used=True,
        )


__all__ = [
    "RUN_DATE",
    "StructuredReasoningAttempt",
    "StructuredReasoningController",
    "StructuredReasoningEmission",
    "build_gemma_structured_reasoning_prompt",
    "build_qwen_structured_reasoning_prompt",
    "default_policy_path",
    "get_repo_root",
    "load_monitorability_policy",
]
