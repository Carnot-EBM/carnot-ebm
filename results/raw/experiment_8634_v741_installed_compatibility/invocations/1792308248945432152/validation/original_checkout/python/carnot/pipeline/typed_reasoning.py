"""Typed reasoning IR for verifier-friendly prompt and response extraction.

Spec: REQ-VERIFY-015, REQ-VERIFY-016, REQ-VERIFY-017,
SCENARIO-VERIFY-015, SCENARIO-VERIFY-016, SCENARIO-VERIFY-017
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_DIRECT_JSON = "direct_json"
_FALLBACK_TEXT = "fallback_text"
_PARSER_VERSION = "20260412"
_JSON_KEYS = {
    "user_constraints",
    "constraints",
    "reasoning_steps",
    "steps",
    "checks",
    "atomic_claims",
    "claims",
    "final_answer",
    "answer",
}


@dataclass
class UserConstraint:
    """Prompt-side constraint represented in typed form."""

    constraint_id: str
    kind: str
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "constraint_id": self.constraint_id,
            "kind": self.kind,
            "text": self.text,
        }


@dataclass
class ReasoningStep:
    """One ordered reasoning step from the model response."""

    step_id: str
    kind: str
    text: str
    claim_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "kind": self.kind,
            "text": self.text,
            "claim_ids": list(self.claim_ids),
        }


@dataclass
class AtomicClaim:
    """A single claim grounded to one reasoning step when available."""

    claim_id: str
    kind: str
    text: str
    value: object | None = None
    step_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "claim_id": self.claim_id,
            "kind": self.kind,
            "text": self.text,
        }
        if self.value is not None:
            data["value"] = self.value
        if self.step_id is not None:
            data["step_id"] = self.step_id
        return data


@dataclass
class FinalAnswer:
    """Normalized final answer for later verifier stages."""

    text: str
    normalized: object | None
    answer_type: str
    source_step_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "text": self.text,
            "answer_type": self.answer_type,
            "normalized": self.normalized,
        }
        if self.source_step_id is not None:
            data["source_step_id"] = self.source_step_id
        return data


@dataclass
class ExtractionProvenance:
    """Records how the IR was extracted."""

    extraction_method: str
    source_format: str
    parser_version: str = _PARSER_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "extraction_method": self.extraction_method,
            "source_format": self.source_format,
            "parser_version": self.parser_version,
        }


@dataclass
class TypedReasoningIR:
    """Typed reasoning graph for deterministic downstream verification."""

    question: str
    user_constraints: list[UserConstraint]
    reasoning_steps: list[ReasoningStep]
    atomic_claims: list[AtomicClaim]
    final_answer: FinalAnswer | None
    provenance: ExtractionProvenance

    def validate(self) -> None:
        """Reject malformed IR deterministically."""
        if self.provenance.extraction_method not in {_DIRECT_JSON, _FALLBACK_TEXT}:
            raise ValueError(f"unknown extraction_method: {self.provenance.extraction_method}")

        step_ids: set[str] = set()
        for step in self.reasoning_steps:
            if step.step_id in step_ids:
                raise ValueError(f"duplicate step_id: {step.step_id}")
            step_ids.add(step.step_id)

        claim_ids: set[str] = set()
        for claim in self.atomic_claims:
            if claim.claim_id in claim_ids:
                raise ValueError(f"duplicate claim_id: {claim.claim_id}")
            claim_ids.add(claim.claim_id)
            if claim.step_id is not None and claim.step_id not in step_ids:
                raise ValueError(f"unknown step_id: {claim.step_id}")
            _ensure_jsonable(claim.value)

        for step in self.reasoning_steps:
            missing = [claim_id for claim_id in step.claim_ids if claim_id not in claim_ids]
            if missing:
                raise ValueError(f"unknown claim_id: {missing[0]}")

        if self.final_answer is not None:
            if (
                self.final_answer.source_step_id is not None
                and self.final_answer.source_step_id not in step_ids
            ):
                raise ValueError(f"unknown step_id: {self.final_answer.source_step_id}")
            _ensure_jsonable(self.final_answer.normalized)

    def to_dict(self) -> dict[str, object]:
        """Serialize the IR as a deterministic dictionary."""
        self.validate()
        data: dict[str, object] = {
            "question": self.question,
            "user_constraints": [constraint.to_dict() for constraint in self.user_constraints],
            "reasoning_steps": [step.to_dict() for step in self.reasoning_steps],
            "atomic_claims": [claim.to_dict() for claim in self.atomic_claims],
            "provenance": self.provenance.to_dict(),
        }
        data["final_answer"] = None if self.final_answer is None else self.final_answer.to_dict()
        return data

    def to_json(self) -> str:
        """Serialize the IR deterministically for downstream verifiers."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TypedReasoningIR:
        """Build a typed IR from a dictionary payload."""
        constraints_raw = _as_dict_list(data.get("user_constraints"))
        steps_raw = _as_dict_list(data.get("reasoning_steps"))
        claims_raw = _as_dict_list(data.get("atomic_claims"))
        final_raw = data.get("final_answer")
        provenance_raw = _as_dict(data.get("provenance"))

        ir = cls(
            question=_as_text(data.get("question")),
            user_constraints=[
                UserConstraint(
                    constraint_id=_as_text(item.get("constraint_id")),
                    kind=_as_text(item.get("kind")),
                    text=_as_text(item.get("text")),
                )
                for item in constraints_raw
            ],
            reasoning_steps=[
                ReasoningStep(
                    step_id=_as_text(item.get("step_id")),
                    kind=_as_text(item.get("kind")),
                    text=_as_text(item.get("text")),
                    claim_ids=_string_list(item.get("claim_ids")),
                )
                for item in steps_raw
            ],
            atomic_claims=[
                AtomicClaim(
                    claim_id=_as_text(item.get("claim_id")),
                    kind=_as_text(item.get("kind")),
                    text=_as_text(item.get("text")),
                    value=item.get("value"),
                    step_id=_optional_text(item.get("step_id")),
                )
                for item in claims_raw
            ],
            final_answer=None
            if final_raw is None
            else FinalAnswer(
                text=_as_text(_as_dict(final_raw).get("text")),
                normalized=_as_dict(final_raw).get("normalized"),
                answer_type=_as_text(_as_dict(final_raw).get("answer_type")),
                source_step_id=_optional_text(_as_dict(final_raw).get("source_step_id")),
            ),
            provenance=ExtractionProvenance(
                extraction_method=_as_text(provenance_raw.get("extraction_method")),
                source_format=_as_text(provenance_raw.get("source_format")),
                parser_version=_as_text(provenance_raw.get("parser_version")),
            ),
        )
        ir.validate()
        return ir

    @classmethod
    def from_json(cls, text: str) -> TypedReasoningIR:
        """Build a typed IR from deterministic JSON."""
        payload = _extract_json_dict(text)
        if payload is None:
            raise ValueError("typed reasoning JSON payload not found")
        return cls.from_dict(payload)


class TypedReasoningExtractor:
    """Dual-path extractor for typed reasoning IR."""

    def __init__(self, parser_version: str = _PARSER_VERSION) -> None:
        self._parser_version = parser_version

    def extract(self, question: str, response: str) -> TypedReasoningIR:
        """Extract typed reasoning from a direct JSON or plain-text response."""
        payload = _extract_json_dict(response)
        if payload is not None and _looks_like_typed_reasoning_payload(payload):
            return self._from_direct_json(question, payload)
        return self._from_fallback_text(question, response)

    def _from_direct_json(self, question: str, payload: dict[str, object]) -> TypedReasoningIR:
        constraints = _coerce_constraints(
            payload.get("user_constraints", payload.get("constraints")),
            question,
        )
        steps = _coerce_steps(
            payload.get(
                "reasoning_steps",
                payload.get("steps", payload.get("checks")),
            )
        )
        claims = _coerce_claims(payload.get("atomic_claims", payload.get("claims")), steps)
        if not claims and steps:
            claims = _claims_from_steps(steps)
        final_answer = _coerce_final_answer(
            payload.get("final_answer", payload.get("answer")),
            steps,
        )
        ir = TypedReasoningIR(
            question=question,
            user_constraints=constraints,
            reasoning_steps=steps,
            atomic_claims=claims,
            final_answer=final_answer,
            provenance=ExtractionProvenance(
                extraction_method=_DIRECT_JSON,
                source_format="json",
                parser_version=self._parser_version,
            ),
        )
        ir.validate()
        return ir

    def _from_fallback_text(self, question: str, response: str) -> TypedReasoningIR:
        constraints = _infer_prompt_constraints(question)
        steps = _fallback_steps(response)
        claims = _claims_from_steps(steps)
        final_answer = _fallback_final_answer(response, steps)
        ir = TypedReasoningIR(
            question=question,
            user_constraints=constraints,
            reasoning_steps=steps,
            atomic_claims=claims,
            final_answer=final_answer,
            provenance=ExtractionProvenance(
                extraction_method=_FALLBACK_TEXT,
                source_format="plain_text",
                parser_version=self._parser_version,
            ),
        )
        ir.validate()
        return ir


def extract_typed_reasoning(question: str, response: str) -> TypedReasoningIR:
    """Convenience helper using the repo's fixed parser version."""
    return TypedReasoningExtractor(parser_version=_PARSER_VERSION).extract(
        question=question,
        response=response,
    )


def _looks_like_typed_reasoning_payload(payload: dict[str, object]) -> bool:
    return any(key in payload for key in _JSON_KEYS)


def _coerce_constraints(raw: object, question: str) -> list[UserConstraint]:
    if raw is None:
        return _infer_prompt_constraints(question)

    items = raw if isinstance(raw, list) else [raw]
    constraints: list[UserConstraint] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            text = _as_text(item.get("text", item.get("description", item.get("constraint"))))
            constraint_id = _optional_text(item.get("constraint_id", item.get("id")))
            constraints.append(
                UserConstraint(
                    constraint_id=constraint_id or f"uc{index}",
                    kind=_optional_text(item.get("kind")) or _constraint_kind(text),
                    text=text,
                )
            )
            continue

        text = _as_text(item)
        constraints.append(
            UserConstraint(
                constraint_id=f"uc{index}",
                kind=_constraint_kind(text),
                text=text,
            )
        )

    return constraints or _infer_prompt_constraints(question)


def _coerce_steps(raw: object) -> list[ReasoningStep]:
    if raw is None:
        return []

    items = raw if isinstance(raw, list) else [raw]
    steps: list[ReasoningStep] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            text = _step_text_from_dict(item)
            step_id = _optional_text(item.get("step_id", item.get("id"))) or f"s{index}"
            claim_ids = _string_list(item.get("claim_ids"))
            steps.append(
                ReasoningStep(
                    step_id=step_id,
                    kind=_optional_text(item.get("kind", item.get("step_type")))
                    or _step_kind(text),
                    text=text,
                    claim_ids=claim_ids,
                )
            )
            continue

        text = _as_text(item)
        steps.append(
            ReasoningStep(
                step_id=f"s{index}",
                kind=_step_kind(text),
                text=text,
            )
        )
    return steps


def _coerce_claims(raw: object, steps: list[ReasoningStep]) -> list[AtomicClaim]:
    if raw is None:
        return []

    items = raw if isinstance(raw, list) else [raw]
    claims: list[AtomicClaim] = []
    for index, item in enumerate(items, start=1):
        default_step_id = steps[index - 1].step_id if index <= len(steps) else None
        if isinstance(item, dict):
            text = _as_text(item.get("text", item.get("description", item.get("claim"))))
            step_id = (
                _optional_text(item.get("step_id", item.get("source_step_id"))) or default_step_id
            )
            claims.append(
                AtomicClaim(
                    claim_id=_optional_text(item.get("claim_id", item.get("id"))) or f"cl{index}",
                    kind=_optional_text(item.get("kind", item.get("claim_type")))
                    or _claim_kind(text),
                    text=text,
                    value=item.get("value"),
                    step_id=step_id,
                )
            )
            continue

        text = _as_text(item)
        claims.append(
            AtomicClaim(
                claim_id=f"cl{index}",
                kind=_claim_kind(text),
                text=text,
                value=_normalized_value(text),
                step_id=default_step_id,
            )
        )

    return claims


def _claims_from_steps(steps: list[ReasoningStep]) -> list[AtomicClaim]:
    claims: list[AtomicClaim] = []
    for index, step in enumerate(steps, start=1):
        claim_id = f"cl{index}"
        step.claim_ids.append(claim_id)
        claims.append(
            AtomicClaim(
                claim_id=claim_id,
                kind=_claim_kind(step.text),
                text=step.text,
                value=_normalized_value(step.text),
                step_id=step.step_id,
            )
        )
    return claims


def _coerce_final_answer(raw: object, steps: list[ReasoningStep]) -> FinalAnswer | None:
    if raw is None:
        return None

    source_step_id = steps[-1].step_id if steps else None
    if isinstance(raw, dict):
        text = _as_text(raw.get("text", raw.get("final", raw.get("value"))))
        normalized = raw.get("normalized", raw.get("value"))
        answer_type = _optional_text(raw.get("answer_type")) or _answer_type(
            normalized if normalized is not None else text
        )
        return FinalAnswer(
            text=text,
            normalized=normalized if normalized is not None else _normalized_value(text),
            answer_type=answer_type,
            source_step_id=_optional_text(raw.get("source_step_id")) or source_step_id,
        )

    text = _as_text(raw)
    return FinalAnswer(
        text=text,
        normalized=_normalized_value(text),
        answer_type=_answer_type(text),
        source_step_id=source_step_id,
    )


def _infer_prompt_constraints(question: str) -> list[UserConstraint]:
    cleaned = question.strip()
    if not cleaned:
        return []

    clauses: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", cleaned):
        sentence = sentence.strip().rstrip(".")
        if not sentence:
            continue
        if " and " in sentence.lower():
            parts = [part.strip() for part in re.split(r"\s+\band\b\s+", sentence)]
            clauses.extend(part for part in parts if part)
        else:
            clauses.append(sentence)

    if not clauses:
        clauses = [cleaned]

    return [
        UserConstraint(
            constraint_id=f"uc{index}",
            kind=_constraint_kind(clause),
            text=clause,
        )
        for index, clause in enumerate(clauses, start=1)
    ]


def _fallback_steps(response: str) -> list[ReasoningStep]:
    reasoning_text = response.strip()
    reasoning_match = re.search(
        r"REASONING:\s*(.*?)(?:\n\s*FINAL\s*:|\Z)",
        reasoning_text,
        re.IGNORECASE | re.DOTALL,
    )
    if reasoning_match:
        reasoning_text = reasoning_match.group(1).strip()

    lines = [
        _clean_reasoning_line(line)
        for line in reasoning_text.splitlines()
        if _clean_reasoning_line(line)
    ]
    if not lines:
        lines = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", response.strip())
            if sentence.strip()
        ]

    return [
        ReasoningStep(
            step_id=f"s{index}",
            kind=_step_kind(line),
            text=line,
        )
        for index, line in enumerate(lines, start=1)
    ]


def _fallback_final_answer(response: str, steps: list[ReasoningStep]) -> FinalAnswer | None:
    final_match = re.search(
        r"FINAL\s*:?\s*(.+)$",
        response.strip(),
        re.IGNORECASE | re.DOTALL,
    )
    if final_match:
        text = final_match.group(1).strip()
    else:
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        text = lines[-1] if lines else response.strip()
        text = _clean_reasoning_line(text)

    if not text:
        return None

    source_step_id = steps[-1].step_id if steps else None
    return FinalAnswer(
        text=text,
        normalized=_normalized_value(text),
        answer_type=_answer_type(text),
        source_step_id=source_step_id,
    )


def _extract_json_dict(text: str) -> dict[str, object] | None:
    stripped = _strip_fence(text).strip()
    decoder = json.JSONDecoder()
    for candidate in (stripped, text.strip()):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed

    for match in re.finditer(r"[{[]", stripped):
        try:
            parsed, _ = decoder.raw_decode(stripped[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _strip_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)```", text.strip(), re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _normalized_value(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, (list, dict)):
        return value

    text = str(value).strip()
    if not text:
        return None

    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if re.fullmatch(r"[-+]?\d+\.\d+", text):
        return float(text)

    payload = _extract_json_dict(text)
    if payload is not None:
        return payload

    number_match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if number_match:
        token = number_match.group(0)
        return float(token) if "." in token else int(token)

    return text


def _answer_type(value: object) -> str:
    normalized = _normalized_value(value)
    if normalized is None:
        return "unknown"
    if isinstance(normalized, bool):
        return "boolean"
    if isinstance(normalized, (int, float)):
        return "number"
    if isinstance(normalized, list):
        return "list"
    if isinstance(normalized, dict):
        return "json_object"
    return "text"


def _constraint_kind(text: str) -> str:
    lowered = text.lower()
    if "do not" in lowered or "don't" in lowered:
        return "forbidden_content"
    if "json" in lowered:
        return "structured_output"
    if any(token in lowered for token in ("exactly", "at least", "at most")):
        return "cardinality"
    if any(token in lowered for token in ("include", "mention")):
        return "content_requirement"
    return "prompt_constraint"


def _step_kind(text: str) -> str:
    lowered = text.lower()
    if "=" in text or any(op in text for op in (" + ", " - ", " / ", " * ")):
        return "calculation"
    if lowered.startswith(("return", "final", "therefore", "so ")):
        return "finalization"
    return "analysis"


def _claim_kind(text: str) -> str:
    if "=" in text or any(op in text for op in (" + ", " - ", " / ", " * ")):
        return "equation"
    return "statement"


def _step_text_from_dict(item: dict[str, object]) -> str:
    if "text" in item:
        return _as_text(item["text"])
    if "description" in item:
        return _as_text(item["description"])
    if "constraint" in item and "evidence" in item:
        return f"{_as_text(item['constraint'])}: {_as_text(item['evidence'])}"
    if "evidence" in item:
        return _as_text(item["evidence"])
    if "constraint" in item:
        return _as_text(item["constraint"])
    return ""


def _clean_reasoning_line(line: str) -> str:
    return re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", line.strip())


def _ensure_jsonable(value: object | None) -> None:
    if value is None:
        return
    try:
        json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError as exc:
        raise ValueError(f"value is not JSON-serializable: {value!r}") from exc


def _as_text(value: object) -> str:
    text = str(value).strip()
    if not text or text == "None":
        return ""
    return text


def _optional_text(value: object) -> str | None:
    text = _as_text(value)
    return text or None


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): val for key, val in value.items()}
    return {}


def _as_dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, object]] = []
    for item in value:
        if isinstance(item, dict):
            items.append({str(key): val for key, val in item.items()})
    return items


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
