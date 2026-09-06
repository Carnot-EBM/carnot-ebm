"""Authorize reuse only inside the context supported by earlier evidence.

The decision type has no outcome field. Exact outcomes enter through the
separate validation type only after an authorization receipt exists. This
separation makes accidental same-event or future-outcome access impossible for
callers that use the typed interface.

Spec refs: REQ-SELFLEARN-7069 and SCENARIO-SELFLEARN-7069-*.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields
from enum import StrEnum
import hashlib
import json
import math
from typing import Any

from carnot.learning.constraint_policy_store import PolicyStore, sha256_bytes


JsonDict = dict[str, Any]
EXPERIENCE_SCHEMA = "carnot.context_bound_experience.v1"
DECISION_SCHEMA = "carnot.context_authorization_decision.v1"
VALIDATION_SCHEMA = "carnot.bounded_validation_event.v1"
MAX_SUPPORTED_UNCERTAINTY_WIDTH = 0.25
SHA256_PREFIX = "sha256:"
FORBIDDEN_DECISION_KEYS = frozenset(
    {
        "current_outcome",
        "exact_outcome",
        "later_outcome",
        "future_outcome",
        "future_outcomes",
        "future_event",
        "future_events",
        "held_group",
        "held_group_label",
        "held_label",
        "split",
        "post_event_aggregate",
        "post_event_aggregates",
        "outcome_provider",
        "outcome_source",
        "mutable_outcome_source",
    }
)


def canonical_bytes(value: Any) -> bytes:
    """Serialize evidence with stable bytes so hashes do not depend on key order."""

    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Hash canonical JSON and name the digest algorithm in the value."""

    return SHA256_PREFIX + hashlib.sha256(canonical_bytes(value)).hexdigest()


def _is_sha256(value: str) -> bool:
    """Accept only the explicit hash shape used by repository evidence."""

    if not value.startswith(SHA256_PREFIX) or len(value) != len(SHA256_PREFIX) + 64:
        return False
    try:
        int(value[len(SHA256_PREFIX) :], 16)
    except ValueError:
        return False
    return True


def _forbidden_paths(value: Any, prefix: str = "") -> tuple[str, ...]:
    """Find denied field names without treating harmless string values as keys."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).casefold() in FORBIDDEN_DECISION_KEYS:
                found.append(path)
            found.extend(_forbidden_paths(nested, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            path = f"{prefix}[{index}]"
            found.extend(_forbidden_paths(nested, path))
    return tuple(found)


@dataclass(frozen=True)
class SupportInterval:
    """Name the inclusive event-time range supported by one observation."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.end) is not int or self.start > self.end:
            raise ValueError("support interval must use ordered integer bounds")

    def contains(self, event_time: int) -> bool:
        """Return true only while the registered support remains current."""

        return self.start <= event_time <= self.end

    def to_dict(self) -> JsonDict:
        """Return the stable JSON representation used by experience receipts."""

        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class NumericInterval:
    """Keep an observed effect as an ordered finite interval."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.lower, bool)
            or isinstance(self.upper, bool)
            or not math.isfinite(float(self.lower))
            or not math.isfinite(float(self.upper))
            or float(self.lower) > float(self.upper)
        ):
            raise ValueError("effect interval must use ordered finite bounds")
        object.__setattr__(self, "lower", float(self.lower))
        object.__setattr__(self, "upper", float(self.upper))

    @property
    def width(self) -> float:
        """Measure uncertainty without opening any later outcome."""

        return self.upper - self.lower

    def to_dict(self) -> JsonDict:
        """Return the stable JSON representation used by experience receipts."""

        return {"lower": self.lower, "upper": self.upper}


class RetentionResult(StrEnum):
    """Report whether exact protected checks survived the observed effect."""

    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ContextBoundExperience:
    """Bind a measured effect to the exact context where it was observed."""

    experience_id: str
    parent_policy_hash: str
    source_group: str
    constraint_schema_hash: str
    support_interval: SupportInterval
    observed_effect_interval: NumericInterval
    retention_result: RetentionResult
    conflicts: tuple[str, ...]
    event_time: int
    source_receipt_hash: str

    def __post_init__(self) -> None:
        if not self.experience_id or not self.source_group:
            raise ValueError("experience identity and source group are required")
        if not _is_sha256(self.parent_policy_hash):
            raise ValueError("parent policy hash must be a prefixed SHA-256 digest")
        if not _is_sha256(self.constraint_schema_hash):
            raise ValueError("constraint schema hash must be a prefixed SHA-256 digest")
        if not _is_sha256(self.source_receipt_hash):
            raise ValueError("source receipt hash must be a prefixed SHA-256 digest")
        if type(self.event_time) is not int or self.event_time < 0:
            raise ValueError("event time must be a non-negative integer")
        if not isinstance(self.support_interval, SupportInterval):
            raise ValueError("support interval must be typed")
        if not isinstance(self.observed_effect_interval, NumericInterval):
            raise ValueError("effect interval must be typed")
        if not isinstance(self.retention_result, RetentionResult):
            raise ValueError("retention result must be typed")
        normalized = tuple(sorted({str(item) for item in self.conflicts if str(item)}))
        object.__setattr__(self, "conflicts", normalized)

    @property
    def canonical_hash(self) -> str:
        """Bind every immutable field to one content receipt."""

        return sha256_json(self.to_dict())

    def to_dict(self) -> JsonDict:
        """Serialize every required fact without adding mutable annotations."""

        return {
            "schema": EXPERIENCE_SCHEMA,
            "experience_id": self.experience_id,
            "parent_policy_hash": self.parent_policy_hash,
            "source_group": self.source_group,
            "constraint_schema_hash": self.constraint_schema_hash,
            "support_interval": self.support_interval.to_dict(),
            "observed_effect_interval": self.observed_effect_interval.to_dict(),
            "retention_result": self.retention_result.value,
            "conflicts": list(self.conflicts),
            "event_time": self.event_time,
            "source_receipt_hash": self.source_receipt_hash,
        }

    def to_json(self) -> str:
        """Return canonical text for durable equality checks."""

        return canonical_bytes(self.to_dict()).decode("utf-8")

    def to_policy_record(self) -> JsonDict:
        """Wrap the immutable record for the existing transactional store."""

        return {
            "policy_key": f"context_experience:{self.experience_id}",
            "protected": False,
            "experience": self.to_dict(),
            "experience_hash": self.canonical_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ContextBoundExperience:
        """Parse a closed schema so unreviewed future fields cannot gain authority."""

        expected = {
            "schema",
            "experience_id",
            "parent_policy_hash",
            "source_group",
            "constraint_schema_hash",
            "support_interval",
            "observed_effect_interval",
            "retention_result",
            "conflicts",
            "event_time",
            "source_receipt_hash",
        }
        unknown = sorted(set(payload) - expected)
        if unknown:
            raise ValueError(f"unknown experience fields: {unknown}")
        if payload.get("schema") != EXPERIENCE_SCHEMA:
            raise ValueError("experience schema mismatch")
        support = payload.get("support_interval")
        effect = payload.get("observed_effect_interval")
        if not isinstance(support, Mapping) or set(support) != {"start", "end"}:
            raise ValueError("support interval shape is invalid")
        if not isinstance(effect, Mapping) or set(effect) != {"lower", "upper"}:
            raise ValueError("effect interval shape is invalid")
        conflicts = payload.get("conflicts")
        if not isinstance(conflicts, list):
            raise ValueError("experience conflicts must be a list")
        try:
            retention = RetentionResult(str(payload.get("retention_result")))
        except ValueError as exc:
            raise ValueError("retention result is invalid") from exc
        return cls(
            experience_id=str(payload.get("experience_id") or ""),
            parent_policy_hash=str(payload.get("parent_policy_hash") or ""),
            source_group=str(payload.get("source_group") or ""),
            constraint_schema_hash=str(payload.get("constraint_schema_hash") or ""),
            support_interval=SupportInterval(int(support["start"]), int(support["end"])),
            observed_effect_interval=NumericInterval(
                float(effect["lower"]), float(effect["upper"])
            ),
            retention_result=retention,
            conflicts=tuple(str(item) for item in conflicts),
            event_time=int(payload.get("event_time", -1)),
            source_receipt_hash=str(payload.get("source_receipt_hash") or ""),
        )

    @classmethod
    def from_json(cls, payload: str) -> ContextBoundExperience:
        """Parse one JSON object through the same closed schema."""

        raw = json.loads(payload)
        if not isinstance(raw, Mapping):
            raise ValueError("experience JSON must contain one object")
        return cls.from_dict(raw)

    @classmethod
    def from_policy_record(cls, payload: Mapping[str, Any]) -> ContextBoundExperience:
        """Verify the transactional wrapper before returning its typed record."""

        experience = payload.get("experience")
        if not isinstance(experience, Mapping):
            raise ValueError("policy record has no typed experience")
        record = cls.from_dict(experience)
        if payload.get("experience_hash") != record.canonical_hash:
            raise ValueError("policy record experience hash mismatch")
        if payload.get("policy_key") != f"context_experience:{record.experience_id}":
            raise ValueError("policy record key mismatch")
        return record


@dataclass(frozen=True)
class DecisionView:
    """Expose only pre-outcome context facts to the authorization machine."""

    event_id: str
    event_time: int
    parent_policy_hash: str
    source_group: str
    related_source_groups: tuple[str, ...]
    constraint_schema_hash: str
    active_constraints: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.event_id or not self.source_group:
            raise ValueError("decision event and source group are required")
        if type(self.event_time) is not int or self.event_time < 0:
            raise ValueError("decision event time must be a non-negative integer")
        if not _is_sha256(self.parent_policy_hash) or not _is_sha256(self.constraint_schema_hash):
            raise ValueError("decision policy and schema require prefixed SHA-256 hashes")
        related = tuple(sorted({str(item) for item in self.related_source_groups if str(item)}))
        active = tuple(sorted({str(item) for item in self.active_constraints if str(item)}))
        object.__setattr__(self, "related_source_groups", related)
        object.__setattr__(self, "active_constraints", active)

    @classmethod
    def schema(cls) -> JsonDict:
        """Publish the exact allowlist used by the time firewall."""

        return {
            "schema": "carnot.context_authorization_view.v1",
            "fields": [item.name for item in fields(cls)],
            "additional_fields_allowed": False,
            "forbidden_fields": sorted(FORBIDDEN_DECISION_KEYS),
        }

    def to_dict(self) -> JsonDict:
        """Serialize the closed decision surface for audit rows."""

        return {
            "event_id": self.event_id,
            "event_time": self.event_time,
            "parent_policy_hash": self.parent_policy_hash,
            "source_group": self.source_group,
            "related_source_groups": list(self.related_source_groups),
            "constraint_schema_hash": self.constraint_schema_hash,
            "active_constraints": list(self.active_constraints),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DecisionView:
        """Reject denied fields at any depth before normal schema parsing."""

        forbidden = _forbidden_paths(payload)
        if forbidden:
            raise ValueError(f"forbidden decision fields: {list(forbidden)}")
        expected = set(cls.schema()["fields"])
        unknown = sorted(set(payload) - expected)
        if unknown:
            raise ValueError(f"unknown decision fields: {unknown}")
        missing = sorted(expected - set(payload))
        if missing:
            raise ValueError(f"missing decision fields: {missing}")
        related = payload["related_source_groups"]
        active = payload["active_constraints"]
        if not isinstance(related, list) or not isinstance(active, list):
            raise ValueError("decision group and constraint fields must be lists")
        return cls(
            event_id=str(payload["event_id"]),
            event_time=int(payload["event_time"]),
            parent_policy_hash=str(payload["parent_policy_hash"]),
            source_group=str(payload["source_group"]),
            related_source_groups=tuple(str(item) for item in related),
            constraint_schema_hash=str(payload["constraint_schema_hash"]),
            active_constraints=tuple(str(item) for item in active),
        )


class Authorization(StrEnum):
    """Limit every pre-outcome result to the three contract decisions."""

    USE = "use"
    VALIDATE = "validate"
    REJECT = "reject"


@dataclass(frozen=True)
class AuthorizationDecision:
    """Bind one deterministic decision to predecessor evidence IDs."""

    authorization: Authorization
    reason: str
    evidence_ids: tuple[str, ...]
    decided_at: int

    def __post_init__(self) -> None:
        if not isinstance(self.authorization, Authorization) or not self.reason:
            raise ValueError("authorization and reason are required")
        if type(self.decided_at) is not int or self.decided_at < 0:
            raise ValueError("decision time must be a non-negative integer")
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))

    def to_dict(self) -> JsonDict:
        """Return the audit row without adding any outcome field."""

        return {
            "schema": DECISION_SCHEMA,
            "authorization": self.authorization.value,
            "reason": self.reason,
            "evidence_ids": list(self.evidence_ids),
            "decided_at": self.decided_at,
        }


class LifecycleState(StrEnum):
    """Name every state reachable under the registered transition table."""

    START = "start"
    USED = "used"
    VALIDATING = "validating"
    REJECTED = "rejected"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    NO_OP = "no_op"


_TRANSITIONS = {
    (LifecycleState.START, "use"): LifecycleState.USED,
    (LifecycleState.START, "validate"): LifecycleState.VALIDATING,
    (LifecycleState.START, "reject"): LifecycleState.REJECTED,
    (LifecycleState.VALIDATING, "commit"): LifecycleState.COMMITTED,
    (LifecycleState.VALIDATING, "rollback"): LifecycleState.ROLLED_BACK,
    (LifecycleState.VALIDATING, "no_op"): LifecycleState.NO_OP,
}


class ContextAuthorizationMachine:
    """Apply a closed rule table to immutable predecessor-only evidence."""

    def __init__(self, *, state: LifecycleState = LifecycleState.START) -> None:
        self.state = state

    @staticmethod
    def transition_table() -> dict[tuple[LifecycleState, str], LifecycleState]:
        """Return a copy so callers cannot change the registered state graph."""

        return dict(_TRANSITIONS)

    def advance(self, action: str) -> LifecycleState:
        """Move only when the source state and action are explicitly registered."""

        key = (self.state, str(action))
        if key not in _TRANSITIONS:
            raise ValueError(f"illegal authorization transition: {self.state.value}:{action}")
        self.state = _TRANSITIONS[key]
        return self.state

    def authorize(
        self,
        records: Sequence[ContextBoundExperience],
        view: DecisionView,
    ) -> AuthorizationDecision:
        """Choose use, validate, or reject without receiving an outcome channel."""

        evidence = tuple(records)
        if not evidence:
            return self._decision(Authorization.REJECT, "missing_facts", (), view)
        if any(record.event_time >= view.event_time for record in evidence):
            return self._decision(
                Authorization.REJECT,
                "non_predecessor_evidence",
                evidence,
                view,
            )
        policy_matches = tuple(
            record for record in evidence if record.parent_policy_hash == view.parent_policy_hash
        )
        if not policy_matches:
            return self._decision(Authorization.REJECT, "policy_mismatch", evidence, view)
        schema_matches = tuple(
            record
            for record in policy_matches
            if record.constraint_schema_hash == view.constraint_schema_hash
        )
        if not schema_matches:
            return self._decision(
                Authorization.REJECT,
                "schema_mismatch",
                policy_matches,
                view,
            )
        direct = tuple(
            record for record in schema_matches if record.source_group == view.source_group
        )
        related = tuple(
            record for record in schema_matches if record.source_group in view.related_source_groups
        )
        candidates = direct or related
        if not candidates:
            return self._decision(Authorization.REJECT, "unknown_context", schema_matches, view)
        candidates = tuple(
            sorted(candidates, key=lambda item: (-item.event_time, item.experience_id))
        )
        active = set(view.active_constraints)
        if any(active.intersection(record.conflicts) for record in candidates):
            return self._decision(Authorization.REJECT, "conflict_overlap", candidates, view)
        current = tuple(
            record for record in candidates if record.support_interval.contains(view.event_time)
        )
        if not current:
            return self._decision(Authorization.REJECT, "stale_support", candidates, view)
        if any(record.retention_result is RetentionResult.FAILED for record in current):
            return self._decision(
                Authorization.REJECT,
                "contradicted_retention",
                current,
                view,
            )
        if any(record.observed_effect_interval.upper <= 0 for record in current):
            return self._decision(
                Authorization.REJECT,
                "contradicted_effect",
                current,
                view,
            )
        if any(
            record.retention_result is not RetentionResult.PASSED
            or record.observed_effect_interval.lower <= 0
            or record.observed_effect_interval.width > MAX_SUPPORTED_UNCERTAINTY_WIDTH
            for record in current
        ):
            return self._decision(
                Authorization.REJECT,
                "unsupported_uncertainty",
                current,
                view,
            )
        if direct:
            return self._decision(Authorization.USE, "direct_context_supported", current, view)
        return self._decision(
            Authorization.VALIDATE,
            "related_context_bounded_uncertainty",
            current,
            view,
        )

    @staticmethod
    def _decision(
        authorization: Authorization,
        reason: str,
        records: Sequence[ContextBoundExperience],
        view: DecisionView,
    ) -> AuthorizationDecision:
        """Create one stable receipt from the exact records considered."""

        return AuthorizationDecision(
            authorization=authorization,
            reason=reason,
            evidence_ids=tuple(record.experience_id for record in records),
            decided_at=view.event_time,
        )


class ValidationDisposition(StrEnum):
    """Name the only post-outcome actions allowed by a validation plan."""

    COMMIT = "commit"
    ROLLBACK = "rollback"
    NO_OP = "no_op"


@dataclass(frozen=True)
class ValidationPlan:
    """Freeze cost, bounds, and the exact decision rule before validation."""

    validation_id: str
    cost: float
    max_abs_outcome: float
    precommitted_decision_rule: str = (
        "commit_if_outcome_gt_cost;rollback_if_outcome_lt_negative_cost;otherwise_no_op"
    )

    def __post_init__(self) -> None:
        cost = float(self.cost)
        bound = float(self.max_abs_outcome)
        if not self.validation_id or not math.isfinite(cost) or cost < 0:
            raise ValueError("validation cost must be finite and non-negative")
        if not math.isfinite(bound) or bound <= 0 or cost > bound:
            raise ValueError("validation bound must be finite and cover the cost")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "max_abs_outcome", bound)

    def disposition_for(self, exact_later_outcome: float) -> ValidationDisposition:
        """Apply the frozen rule after the exact later outcome becomes visible."""

        outcome = float(exact_later_outcome)
        if not math.isfinite(outcome) or abs(outcome) > self.max_abs_outcome:
            raise ValueError("exact outcome is outside the validation bound")
        if outcome > self.cost:
            return ValidationDisposition.COMMIT
        if outcome < -self.cost:
            return ValidationDisposition.ROLLBACK
        return ValidationDisposition.NO_OP

    def to_dict(self) -> JsonDict:
        """Publish the precommit without attaching the later outcome."""

        return {
            "validation_id": self.validation_id,
            "cost": self.cost,
            "max_abs_outcome": self.max_abs_outcome,
            "precommitted_decision_rule": self.precommitted_decision_rule,
        }


@dataclass(frozen=True)
class BoundedValidationEvent:
    """Join a prior validate receipt to an exact outcome only after decision time."""

    plan: ValidationPlan
    authorization_decision: AuthorizationDecision
    exact_later_outcome: float
    outcome_receipt_hash: str
    outcome_opened_at: int
    disposition: ValidationDisposition

    @classmethod
    def open_after_decision(
        cls,
        *,
        plan: ValidationPlan,
        authorization_decision: AuthorizationDecision,
        exact_later_outcome: float,
        outcome_receipt_hash: str,
        outcome_opened_at: int,
    ) -> BoundedValidationEvent:
        """Open sealed evidence only for a prior validate decision."""

        if authorization_decision.authorization is not Authorization.VALIDATE:
            raise ValueError("bounded validation requires a validate decision")
        if outcome_opened_at <= authorization_decision.decided_at:
            raise ValueError("exact outcome must open after authorization")
        if not _is_sha256(outcome_receipt_hash):
            raise ValueError("outcome receipt must be a prefixed SHA-256 digest")
        disposition = plan.disposition_for(exact_later_outcome)
        return cls(
            plan=plan,
            authorization_decision=authorization_decision,
            exact_later_outcome=float(exact_later_outcome),
            outcome_receipt_hash=outcome_receipt_hash,
            outcome_opened_at=int(outcome_opened_at),
            disposition=disposition,
        )

    @classmethod
    def schema(cls) -> JsonDict:
        """Describe the post-decision event fields for artifact consumers."""

        return {
            "schema": VALIDATION_SCHEMA,
            "fields": [item.name for item in fields(cls)],
            "outcome_access": "post_decision_only",
            "terminal_actions": [item.value for item in ValidationDisposition],
        }


def settle_validation(
    store: PolicyStore,
    event: BoundedValidationEvent,
    record: ContextBoundExperience,
) -> JsonDict:
    """Apply the precommitted action with atomic commit or exact rollback."""

    if event.disposition is ValidationDisposition.NO_OP:
        state_hash = store.state_hash
        return {
            "action": "no_op",
            "reason": "no_preference_supported",
            "state_hash": state_hash,
            "state_unchanged": store.state_hash == state_hash,
        }

    parent = store.state_bytes
    parent_hash = sha256_bytes(parent)
    commit = store.commit(record.to_policy_record())
    if event.disposition is ValidationDisposition.COMMIT:
        return {"action": "commit", **deepcopy(commit)}
    rollback = store.rollback(
        parent,
        transaction_id=str(commit["transaction_id"]),
        reason="harmful_validation_outcome",
    )
    return {
        "action": "rollback",
        "committed": True,
        "parent_state_hash": parent_hash,
        "candidate_state_hash": commit["new_state_hash"],
        **deepcopy(rollback),
    }


__all__ = [
    "Authorization",
    "AuthorizationDecision",
    "BoundedValidationEvent",
    "ContextAuthorizationMachine",
    "ContextBoundExperience",
    "DECISION_SCHEMA",
    "DecisionView",
    "EXPERIENCE_SCHEMA",
    "FORBIDDEN_DECISION_KEYS",
    "LifecycleState",
    "MAX_SUPPORTED_UNCERTAINTY_WIDTH",
    "NumericInterval",
    "RetentionResult",
    "SupportInterval",
    "VALIDATION_SCHEMA",
    "ValidationDisposition",
    "ValidationPlan",
    "canonical_bytes",
    "settle_validation",
    "sha256_json",
]
