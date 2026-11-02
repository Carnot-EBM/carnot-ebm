"""Fast/slow continual-learning facade for FR-11 retention experiments.

The training helper in ``carnot.training.fast_slow`` is focused on prompt
repair.  This module exposes the smaller experiment-facing interface requested
by FR-11: slow verifier constraints stay shared and stable, while fast verifier
context is updated online from verified examples and can be cleared between
queries without mutating the slow weights.

Spec: REQ-LEARN-2357, SCENARIO-LEARN-2357.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


BASE_SLOW_CONSTRAINTS: dict[str, float] = {
    "parse:numbers": 1.0,
    "parse:boolean_literals": 1.0,
    "normalize:answer": 1.0,
}


@dataclass
class VerifiedConstraint:
    """One verifier-approved constraint retained in the fast context cache."""

    constraint_id: str
    description: str
    support: int = 0
    domains_seen: set[str] = field(default_factory=set)
    example_queries: list[str] = field(default_factory=list)

    def observe(self, *, domain: str, query: str) -> None:
        self.support += 1
        if domain:
            self.domains_seen.add(domain)
        if query and len(self.example_queries) < 3:
            self.example_queries.append(" ".join(query.split())[:180])

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "description": self.description,
            "support": self.support,
            "domains_seen": sorted(self.domains_seen),
            "example_queries": list(self.example_queries),
        }


@dataclass
class SlowWeights:
    """Verifier-initialized constraints shared across all domains.

    These are the experiment's slow weights: they represent stable verifier
    abilities such as parsing numbers and normalizing answers.  Online learning
    writes go to ``FastWeights`` so this mapping can be snapshotted and checked
    for non-mutation during the retention protocol.
    """

    constraint_weights: dict[str, float] = field(
        default_factory=lambda: dict(BASE_SLOW_CONSTRAINTS)
    )
    verifier: Any | None = None

    @classmethod
    def from_verifier(cls, verifier: Any | None) -> SlowWeights:
        weights = dict(BASE_SLOW_CONSTRAINTS)
        verifier_weights = getattr(verifier, "constraint_weights", None)
        if isinstance(verifier_weights, Mapping):
            for key, value in verifier_weights.items():
                try:
                    weights[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        return cls(constraint_weights=weights, verifier=verifier)

    def active_constraints(self) -> set[str]:
        return {key for key, value in self.constraint_weights.items() if value > 0.0}


@dataclass
class FastWeights:
    """Verifier context updated online from successful checks.

    ``query_context`` is the per-query scratch cache and is safe to clear after
    each query.  ``cache`` keeps the verified fast constraints that the retention
    experiment measures across domain phases.
    """

    cache: dict[str, VerifiedConstraint] = field(default_factory=dict)
    query_context: dict[str, VerifiedConstraint] = field(default_factory=dict)
    update_count: int = 0

    def update(
        self,
        *,
        query: str,
        domain: str,
        constraints: Iterable[Any],
    ) -> list[VerifiedConstraint]:
        observed: list[VerifiedConstraint] = []
        self.query_context.clear()
        for raw_constraint in constraints:
            constraint_id, description = _normalize_constraint(raw_constraint)
            if not constraint_id:
                continue
            constraint = self.cache.get(constraint_id)
            if constraint is None:
                constraint = VerifiedConstraint(
                    constraint_id=constraint_id,
                    description=description,
                )
                self.cache[constraint_id] = constraint
            constraint.observe(domain=domain, query=query)
            self.query_context[constraint_id] = constraint
            observed.append(constraint)
        self.update_count += len(observed)
        return observed

    def clear_query_context(self) -> None:
        self.query_context.clear()

    def clear(self) -> None:
        self.cache.clear()
        self.query_context.clear()
        self.update_count = 0

    def active_constraints(self) -> set[str]:
        return set(self.cache)

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_count": self.update_count,
            "cache_size": len(self.cache),
            "query_context_size": len(self.query_context),
            "constraints": [constraint.to_dict() for constraint in self.cache.values()],
        }


@dataclass
class FastSlowTrainer:
    """Small FST trainer with slow verifier weights and fast context updates."""

    slow_weights: SlowWeights = field(default_factory=SlowWeights)
    fast_weights: FastWeights = field(default_factory=FastWeights)

    @classmethod
    def from_pipeline(cls, pipeline: Any) -> FastSlowTrainer:
        verifier = getattr(pipeline, "_and_compose_verifier", None)
        if verifier is None:
            verifier = getattr(pipeline, "verifier", None)
        return cls(slow_weights=SlowWeights.from_verifier(verifier))

    def clear_query_context(self) -> None:
        self.fast_weights.clear_query_context()

    def update_fast(self, query: Any, verification_result: Any) -> list[VerifiedConstraint]:
        """Add verifier-approved constraints to the fast cache.

        Unverified results are ignored so failed checks cannot teach the cache.
        This mirrors the FR-11 safety boundary: continuous self-learning only
        records constraints after a verifier has accepted the observation.
        """

        if not _result_verified(verification_result):
            self.fast_weights.clear_query_context()
            return []
        query_text = _query_text(query)
        domain = _result_field(verification_result, "domain", _infer_domain(query_text))
        constraints = _result_field(
            verification_result,
            "constraints",
            _constraints_for_query(query_text),
        )
        return self.fast_weights.update(
            query=query_text,
            domain=str(domain or ""),
            constraints=_as_iterable_constraints(constraints),
        )

    def predict(self, query: Any) -> str | None:
        """Predict an answer using stable slow constraints plus learned fast ones."""

        query_text = _query_text(query)
        required = _constraints_for_query(query_text)
        active = self.slow_weights.active_constraints() | self.fast_weights.active_constraints()
        if not required.issubset(active):
            return None
        return _solve_query(query_text)

    def certificate(self) -> dict[str, Any]:
        cross_domain = [
            constraint.to_dict()
            for constraint in self.fast_weights.cache.values()
            if len(constraint.domains_seen) > 1
        ]
        return {
            "enabled": True,
            "slow_constraint_count": len(self.slow_weights.constraint_weights),
            "slow_constraints": dict(sorted(self.slow_weights.constraint_weights.items())),
            "fast_update_count": self.fast_weights.update_count,
            "fast_cache_size": len(self.fast_weights.cache),
            "query_context_size": len(self.fast_weights.query_context),
            "cross_domain_constraints": cross_domain,
        }


def _query_text(query: Any) -> str:
    if isinstance(query, str):
        return query
    if isinstance(query, Mapping):
        return str(query.get("question") or query.get("query") or query)
    return str(getattr(query, "question", query))


def _result_field(result: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(field_name, default)
    return getattr(result, field_name, default)


def _result_verified(result: Any) -> bool:
    if _result_field(result, "verified", None) is not None:
        return bool(_result_field(result, "verified"))
    if _result_field(result, "is_correct", None) is not None:
        return bool(_result_field(result, "is_correct"))
    return False


def _as_iterable_constraints(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str | bytes):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _normalize_constraint(raw_constraint: Any) -> tuple[str, str]:
    if isinstance(raw_constraint, Mapping):
        constraint_id = str(
            raw_constraint.get("id")
            or raw_constraint.get("constraint_id")
            or raw_constraint.get("constraint_type")
            or raw_constraint.get("name")
            or ""
        )
        description = str(raw_constraint.get("description") or constraint_id)
        return constraint_id, description
    constraint_id = str(
        getattr(raw_constraint, "constraint_id", "")
        or getattr(raw_constraint, "constraint_type", "")
        or raw_constraint
    )
    description = str(getattr(raw_constraint, "description", "") or constraint_id)
    return constraint_id, description


def _infer_domain(query_text: str) -> str:
    text = query_text.lower()
    if "def " in text or "return" in text:
        return "code"
    if "true" in text or "false" in text:
        return "logic"
    return "arithmetic"


def _constraints_for_query(query_text: str) -> set[str]:
    text = query_text.lower()
    if "def " in text or "return" in text:
        return {"operation:addition", "syntax:python_return_expr"}
    if re.search(r"\b(true|false)\b\s+(and|or|xor)\s+\b(true|false)\b", text):
        return {"logic:boolean_algebra"}
    if "add" in text or "+" in text:
        return {"operation:addition"}
    return set()


def _solve_query(query_text: str) -> str | None:
    text = query_text.lower()
    code_match = re.search(r"return\s+(-?\d+)\s*\+\s*(-?\d+)", text)
    if code_match:
        return str(int(code_match.group(1)) + int(code_match.group(2)))

    add_match = re.search(r"add\s+(-?\d+)\s+(?:and|to)\s+(-?\d+)", text)
    if add_match:
        return str(int(add_match.group(1)) + int(add_match.group(2)))
    plus_match = re.search(r"(-?\d+)\s*\+\s*(-?\d+)", text)
    if plus_match:
        return str(int(plus_match.group(1)) + int(plus_match.group(2)))

    logic_match = re.search(
        r"\b(true|false)\b\s+(and|or|xor)\s+\b(true|false)\b",
        text,
    )
    if logic_match:
        left = logic_match.group(1) == "true"
        op = logic_match.group(2)
        right = logic_match.group(3) == "true"
        if op == "and":
            value = left and right
        elif op == "or":
            value = left or right
        else:
            value = left != right
        return str(value).lower()
    return None


__all__ = [
    "FastSlowTrainer",
    "FastWeights",
    "SlowWeights",
    "VerifiedConstraint",
]
