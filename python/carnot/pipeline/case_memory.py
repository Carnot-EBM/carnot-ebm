"""Additive case-based memory for more specific live-trace reuse.

Spec: REQ-VERIFY-050, REQ-VERIFY-051,
SCENARIO-VERIFY-052, SCENARIO-VERIFY-053, SCENARIO-VERIFY-054,
SCENARIO-VERIFY-055
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VERSION = 1
_MAX_PROMPT_TOKENS = 6
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "answer",
        "are",
        "as",
        "at",
        "be",
        "because",
        "by",
        "did",
        "does",
        "for",
        "from",
        "if",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "then",
        "to",
        "was",
        "were",
        "with",
        "write",
    }
)
_KNOWN_PROPERTIES = (
    "annotated_return_type",
    "deterministic",
    "input_immutability",
    "no_exception",
    "reverse_output",
    "sorted_output",
)


def _clean_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def _normalise_name(value: str) -> str:
    return _clean_text(value)


def _unique_sorted(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(sorted({value for value in values if value}))


def _merge_unique(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in (*left, *right):
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(text.lower()))


def _prompt_signature(
    prompt_text: str,
    description_texts: tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
    source = _clean_text(prompt_text) if _clean_text(prompt_text) else " ".join(description_texts)
    tokens: list[str] = []
    seen: set[str] = set()
    for token in _tokenize(source):
        if token.isdigit() or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= _MAX_PROMPT_TOKENS:
            break
    if not tokens:
        return "generic", ()
    return "|".join(tokens), tuple(tokens)


def _violation_families(violation_types: tuple[str, ...]) -> tuple[str, ...]:
    families: list[str] = []
    for violation_type in violation_types:
        families.append(violation_type.split(":", 1)[0])
    return _unique_sorted(families)


def _infer_property_names(
    explicit_properties: tuple[str, ...],
    *,
    violation_types: tuple[str, ...],
    description_texts: tuple[str, ...],
) -> tuple[str, ...]:
    combined = " ".join((*violation_types, *description_texts)).lower()
    found = {property_name for property_name in _KNOWN_PROPERTIES if property_name in combined}
    return _unique_sorted([*explicit_properties, *found])


def _repair_outcome(
    *,
    baseline_success: bool | None,
    repair_success: bool | None,
) -> str:
    if baseline_success is None or repair_success is None:
        return "unknown"
    if repair_success and not baseline_success:
        return "improved"
    if baseline_success and not repair_success:
        return "regressed"
    if baseline_success and repair_success:
        return "unchanged_success"
    return "unchanged_failure"


def _case_kind(
    *,
    benchmark: str,
    benchmark_slice: str,
    property_names: tuple[str, ...],
    violation_families: tuple[str, ...],
) -> str:
    lowered_slice = benchmark_slice.lower()
    lowered_benchmark = benchmark.lower()
    if "humaneval" in lowered_benchmark or "code" in lowered_slice or property_names:
        return "code_verification"
    if "gsm8k" in lowered_benchmark or "semantic" in lowered_slice:
        return "semantic_verification"
    if any("semantic" in family for family in violation_families):
        return "semantic_verification"
    return "constraint_verification"


@dataclass(frozen=True)
class CaseProvenance:
    """One source trace that contributed support to a stored case."""

    source_experiment: int | None
    case_id: str
    model_name: str
    benchmark: str
    source_artifact: str | None = None
    response_mode: str = ""
    verifier_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_experiment": self.source_experiment,
            "case_id": self.case_id,
            "model_name": self.model_name,
            "benchmark": self.benchmark,
            "source_artifact": self.source_artifact,
            "response_mode": self.response_mode,
            "verifier_path": self.verifier_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CaseProvenance:
        source_experiment = payload.get("source_experiment")
        source_artifact = payload.get("source_artifact")
        return cls(
            source_experiment=int(source_experiment) if source_experiment is not None else None,
            case_id=str(payload.get("case_id") or ""),
            model_name=str(payload.get("model_name") or ""),
            benchmark=str(payload.get("benchmark") or ""),
            source_artifact=str(source_artifact) if source_artifact is not None else None,
            response_mode=str(payload.get("response_mode") or ""),
            verifier_path=str(payload.get("verifier_path") or ""),
        )


@dataclass(frozen=True)
class CaseKey:
    """Deterministic retrieval key for one aggregated case."""

    model_name: str
    benchmark_slice: str
    violation_families: tuple[str, ...]
    prompt_sketch: str
    property_names: tuple[str, ...]
    repair_outcome: str

    @property
    def fingerprint(self) -> str:
        return " | ".join(
            [
                self.model_name,
                self.benchmark_slice,
                ",".join(self.violation_families) or "-",
                self.prompt_sketch,
                ",".join(self.property_names) or "-",
                self.repair_outcome,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "benchmark_slice": self.benchmark_slice,
            "violation_families": list(self.violation_families),
            "prompt_sketch": self.prompt_sketch,
            "property_names": list(self.property_names),
            "repair_outcome": self.repair_outcome,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CaseKey:
        return cls(
            model_name=str(payload.get("model_name") or ""),
            benchmark_slice=str(payload.get("benchmark_slice") or ""),
            violation_families=tuple(str(item) for item in payload.get("violation_families", [])),
            prompt_sketch=str(payload.get("prompt_sketch") or "generic"),
            property_names=tuple(str(item) for item in payload.get("property_names", [])),
            repair_outcome=str(payload.get("repair_outcome") or "unknown"),
        )


@dataclass(frozen=True)
class CaseRecord:
    """Normalized one-trace case prior to aggregation."""

    case_kind: str
    model_name: str
    benchmark: str
    benchmark_slice: str
    violation_types: tuple[str, ...]
    violation_families: tuple[str, ...]
    prompt_sketch: str
    prompt_tokens: tuple[str, ...]
    property_names: tuple[str, ...]
    repair_outcome: str
    confidence: float
    provenance: CaseProvenance

    @classmethod
    def normalize(
        cls,
        *,
        benchmark: str,
        benchmark_slice: str,
        model_name: str,
        case_id: str,
        violation_types: tuple[str, ...] | list[str],
        prompt_text: str = "",
        description_texts: tuple[str, ...] | list[str] = (),
        property_names: tuple[str, ...] | list[str] = (),
        baseline_success: bool | None = None,
        repair_success: bool | None = None,
        confidence: float = 0.0,
        source_experiment: int | None = None,
        source_artifact: str | None = None,
        response_mode: str = "",
        verifier_path: str = "",
    ) -> CaseRecord:
        normalized_violation_types = tuple(
            _normalise_name(value) for value in violation_types if _normalise_name(value)
        )
        normalized_description_texts = tuple(
            _clean_text(value) for value in description_texts if _clean_text(value)
        )
        normalized_properties = _infer_property_names(
            tuple(_normalise_name(value) for value in property_names if _normalise_name(value)),
            violation_types=normalized_violation_types,
            description_texts=normalized_description_texts,
        )
        prompt_sketch, prompt_tokens = _prompt_signature(prompt_text, normalized_description_texts)
        violation_families = _violation_families(normalized_violation_types)
        case_kind = _case_kind(
            benchmark=benchmark,
            benchmark_slice=benchmark_slice,
            property_names=normalized_properties,
            violation_families=violation_families,
        )
        return cls(
            case_kind=case_kind,
            model_name=_normalise_name(model_name),
            benchmark=_normalise_name(benchmark),
            benchmark_slice=_normalise_name(benchmark_slice),
            violation_types=normalized_violation_types,
            violation_families=violation_families,
            prompt_sketch=prompt_sketch,
            prompt_tokens=prompt_tokens,
            property_names=normalized_properties,
            repair_outcome=_repair_outcome(
                baseline_success=baseline_success,
                repair_success=repair_success,
            ),
            confidence=max(0.0, min(1.0, float(confidence))),
            provenance=CaseProvenance(
                source_experiment=source_experiment,
                case_id=_normalise_name(case_id),
                model_name=_normalise_name(model_name),
                benchmark=_normalise_name(benchmark),
                source_artifact=_normalise_name(source_artifact) if source_artifact else None,
                response_mode=_normalise_name(response_mode),
                verifier_path=_normalise_name(verifier_path),
            ),
        )

    @property
    def key(self) -> CaseKey:
        return CaseKey(
            model_name=self.model_name,
            benchmark_slice=self.benchmark_slice,
            violation_families=self.violation_families,
            prompt_sketch=self.prompt_sketch,
            property_names=self.property_names,
            repair_outcome=self.repair_outcome,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_kind": self.case_kind,
            "model_name": self.model_name,
            "benchmark": self.benchmark,
            "benchmark_slice": self.benchmark_slice,
            "violation_types": list(self.violation_types),
            "violation_families": list(self.violation_families),
            "prompt_sketch": self.prompt_sketch,
            "prompt_tokens": list(self.prompt_tokens),
            "property_names": list(self.property_names),
            "repair_outcome": self.repair_outcome,
            "confidence": self.confidence,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CaseRecord:
        provenance_payload = payload.get("provenance")
        return cls(
            case_kind=str(payload.get("case_kind") or ""),
            model_name=str(payload.get("model_name") or ""),
            benchmark=str(payload.get("benchmark") or ""),
            benchmark_slice=str(payload.get("benchmark_slice") or ""),
            violation_types=tuple(str(item) for item in payload.get("violation_types", [])),
            violation_families=tuple(str(item) for item in payload.get("violation_families", [])),
            prompt_sketch=str(payload.get("prompt_sketch") or "generic"),
            prompt_tokens=tuple(str(item) for item in payload.get("prompt_tokens", [])),
            property_names=tuple(str(item) for item in payload.get("property_names", [])),
            repair_outcome=str(payload.get("repair_outcome") or "unknown"),
            confidence=float(payload.get("confidence") or 0.0),
            provenance=CaseProvenance.from_dict(
                provenance_payload if isinstance(provenance_payload, dict) else {}
            ),
        )


@dataclass(frozen=True)
class CaseQuery:
    """Retrieval query derived from a normalized case shape."""

    model_name: str
    benchmark_slice: str
    violation_types: tuple[str, ...]
    violation_families: tuple[str, ...]
    prompt_sketch: str
    prompt_tokens: tuple[str, ...]
    property_names: tuple[str, ...]
    preferred_repair_outcome: str | None = None

    @classmethod
    def from_record(
        cls,
        record: CaseRecord,
        *,
        preferred_repair_outcome: str | None = None,
    ) -> CaseQuery:
        return cls(
            model_name=record.model_name,
            benchmark_slice=record.benchmark_slice,
            violation_types=record.violation_types,
            violation_families=record.violation_families,
            prompt_sketch=record.prompt_sketch,
            prompt_tokens=record.prompt_tokens,
            property_names=record.property_names,
            preferred_repair_outcome=preferred_repair_outcome,
        )


@dataclass(frozen=True)
class CaseEntry:
    """Aggregated support for one case key."""

    key: CaseKey
    case_kind: str
    benchmark: str
    violation_types: tuple[str, ...]
    prompt_tokens: tuple[str, ...]
    support: int
    confidence: float
    provenance: tuple[CaseProvenance, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "case_kind": self.case_kind,
            "benchmark": self.benchmark,
            "violation_types": list(self.violation_types),
            "prompt_tokens": list(self.prompt_tokens),
            "support": self.support,
            "confidence": self.confidence,
            "provenance": [item.to_dict() for item in self.provenance],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CaseEntry:
        key_payload = payload.get("key")
        return cls(
            key=CaseKey.from_dict(key_payload if isinstance(key_payload, dict) else {}),
            case_kind=str(payload.get("case_kind") or ""),
            benchmark=str(payload.get("benchmark") or ""),
            violation_types=tuple(str(item) for item in payload.get("violation_types", [])),
            prompt_tokens=tuple(str(item) for item in payload.get("prompt_tokens", [])),
            support=int(payload.get("support") or 0),
            confidence=float(payload.get("confidence") or 0.0),
            provenance=tuple(
                CaseProvenance.from_dict(item)
                for item in payload.get("provenance", [])
                if isinstance(item, dict)
            ),
        )


@dataclass(frozen=True)
class CaseMatch:
    """Ranked retrieval result for one aggregated case."""

    entry: CaseEntry
    score: int
    matched_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "score": self.score,
            "matched_fields": list(self.matched_fields),
        }


class CaseMemory:
    """Deterministic case-based memory keyed by specific live-trace features."""

    def __init__(self) -> None:
        self._entries: dict[CaseKey, CaseEntry] = {}

    def record(self, record: CaseRecord) -> None:
        existing = self._entries.get(record.key)
        if existing is None:
            self._entries[record.key] = CaseEntry(
                key=record.key,
                case_kind=record.case_kind,
                benchmark=record.benchmark,
                violation_types=record.violation_types,
                prompt_tokens=record.prompt_tokens,
                support=1,
                confidence=record.confidence,
                provenance=(record.provenance,),
            )
            return

        support = existing.support + 1
        updated_provenance = existing.provenance
        if record.provenance not in updated_provenance:
            updated_provenance = (*updated_provenance, record.provenance)
        weighted_confidence = (
            (existing.confidence * existing.support) + record.confidence
        ) / support
        self._entries[record.key] = CaseEntry(
            key=existing.key,
            case_kind=existing.case_kind,
            benchmark=existing.benchmark,
            violation_types=_merge_unique(existing.violation_types, record.violation_types),
            prompt_tokens=_merge_unique(existing.prompt_tokens, record.prompt_tokens),
            support=support,
            confidence=weighted_confidence,
            provenance=updated_provenance,
        )

    def entries(self) -> tuple[CaseEntry, ...]:
        return tuple(
            self._entries[key] for key in sorted(self._entries, key=lambda item: item.fingerprint)
        )

    def _score(self, entry: CaseEntry, query: CaseQuery) -> CaseMatch | None:
        if entry.key.benchmark_slice != query.benchmark_slice:
            return None

        matched_fields: list[str] = []
        score = 0

        if entry.key.model_name == query.model_name:
            matched_fields.append("model_name")
            score += 30

        family_overlap = len(set(entry.key.violation_families) & set(query.violation_families))
        if family_overlap:
            matched_fields.append("violation_families")
            score += 22 * family_overlap

        type_overlap = len(set(entry.violation_types) & set(query.violation_types))
        if type_overlap:
            matched_fields.append("violation_types")
            score += 10 * type_overlap

        property_overlap = len(set(entry.key.property_names) & set(query.property_names))
        if property_overlap:
            matched_fields.append("property_names")
            score += 20 * property_overlap

        prompt_overlap = 0
        if query.prompt_sketch != "generic" and entry.key.prompt_sketch == query.prompt_sketch:
            matched_fields.append("prompt_sketch")
            score += 18
            prompt_overlap = len(set(entry.prompt_tokens) & set(query.prompt_tokens))
        else:
            prompt_overlap = len(set(entry.prompt_tokens) & set(query.prompt_tokens))
            if prompt_overlap:
                matched_fields.append("prompt_tokens")
                score += 4 * prompt_overlap

        if (
            query.preferred_repair_outcome is not None
            and entry.key.repair_outcome == query.preferred_repair_outcome
        ):
            matched_fields.append("repair_outcome")
            score += 12

        meaningful_overlap = (
            family_overlap
            or type_overlap
            or property_overlap
            or prompt_overlap
            or entry.key.model_name == query.model_name
        )
        if not meaningful_overlap:
            return None

        score += min(entry.support, 10)
        score += int(round(entry.confidence * 10))
        return CaseMatch(entry=entry, score=score, matched_fields=tuple(matched_fields))

    def retrieve(
        self,
        query: CaseQuery,
        *,
        limit: int = 5,
        min_support: int = 1,
    ) -> list[CaseMatch]:
        matches: list[CaseMatch] = []
        for entry in self.entries():
            if entry.support < min_support:
                continue
            match = self._score(entry, query)
            if match is not None:
                matches.append(match)
        matches.sort(
            key=lambda item: (
                -item.score,
                -item.entry.support,
                -item.entry.confidence,
                item.entry.key.fingerprint,
            )
        )
        return matches[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "entries": [entry.to_dict() for entry in self.entries()],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CaseMemory:
        if payload.get("version") != VERSION:
            raise ValueError(f"Unsupported case memory format (expected version={VERSION})")
        memory = cls()
        for raw_entry in payload.get("entries", []):
            if not isinstance(raw_entry, dict):
                continue
            entry = CaseEntry.from_dict(raw_entry)
            memory._entries[entry.key] = entry
        return memory

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> CaseMemory:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Case memory payload must be a JSON object")
        return cls.from_dict(payload)

    def add_trace_selective(
        self,
        record: CaseRecord,
        *,
        violation_energy: float,
        model_confidence: float,
        min_contrast: float = 0.5,
    ) -> bool:
        """Store a trace only when violation_energy and model_confidence disagree.

        **Detailed explanation for engineers:**
            Implements the ATLAS (arXiv 2511.01093) selective consolidation
            strategy: only high-contrast interactions are worth retaining
            because they represent surprising disagreements between the EBM
            and the model's apparent confidence.

            Contrast is defined as ``abs(violation_energy - model_confidence)``.
            The comparison is strict (``>``) so a trace at exactly min_contrast
            is NOT retained.

            This method is purely additive — it does not modify or remove any
            existing entries.  The underlying ``record()`` path is unchanged.

        Args:
            record:            The normalised CaseRecord to potentially store.
            violation_energy:  EBM verification energy (higher = stronger violation).
            model_confidence:  Model's self-reported confidence in [0, 1].
            min_contrast:      Minimum contrast required to retain the trace.
                               Default 0.5 (target: ~40% retention per ATLAS).

        Returns:
            True if the trace was stored, False if discarded as low-contrast.

        Spec: REQ-LEARN-016-3, REQ-LEARN-016-4, SCENARIO-LEARN-028
        """
        contrast = abs(violation_energy - model_confidence)
        if contrast <= min_contrast:
            return False
        self.record(record)
        return True

    def __len__(self) -> int:
        return len(self._entries)


__all__ = [
    "CaseEntry",
    "CaseKey",
    "CaseMatch",
    "CaseMemory",
    "CaseProvenance",
    "CaseQuery",
    "CaseRecord",
    "VERSION",
]
# Note: CaseMemory.add_trace_selective is a method, not a module-level symbol.
