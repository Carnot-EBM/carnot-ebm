"""Verifier-memory promotion helpers for continuous self-learning.

The module keeps verifier memories as controller artifacts, not model-weight
updates. A candidate can enter durable memory only after deterministic guards
pass and held-out evidence clears the promotion threshold; null evidence is
recorded as rollback instead of being silently ignored.

Spec: REQ-LEARN-5214, SCENARIO-LEARN-5214
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping


PROMOTED = "promoted"
HELD = "held"
ROLLED_BACK = "rolled_back"

DEFAULT_PROMOTION_THRESHOLD = 0.02

FORBIDDEN_TEST_GOLD_TOKENS = frozenset(
    {
        "correct_for_eval_only",
        "oracle_target",
        "target_grid",
        "test_gold_grid",
        "test_gold_payload",
        "test_output",
        "z_gold",
    }
)


@dataclass(frozen=True)
class PromotionDecision:
    """Policy result for one verifier-memory candidate."""

    promotion_state: str
    reason: str
    rollback_reason: str | None = None


def decide_promotion(
    *,
    deterministic_guard_result: bool | Mapping[str, Any],
    heldout_delta: float | int | Mapping[str, Any] | None,
    promotion_threshold: float = DEFAULT_PROMOTION_THRESHOLD,
) -> PromotionDecision:
    """Decide whether a verifier-memory candidate is promoted, held, or rolled back.

    Args:
        deterministic_guard_result: Either a boolean guard result or a mapping
            with `passed`, `checks`, and/or `no_test_gold_leak` fields.
        heldout_delta: Numeric held-out delta or a mapping containing `value`.
        promotion_threshold: Minimum positive held-out delta required for
            promotion.

    Returns:
        A PromotionDecision with `promotion_state` in
        {"promoted", "held", "rolled_back"}.
    """
    if not _guard_passed(deterministic_guard_result):
        return PromotionDecision(
            promotion_state=ROLLED_BACK,
            reason="deterministic_guard_failed",
            rollback_reason="deterministic_guard_failed",
        )

    delta = _heldout_delta_value(heldout_delta)
    if delta is None:
        return PromotionDecision(
            promotion_state=ROLLED_BACK,
            reason="heldout_delta_missing",
            rollback_reason="heldout_delta_missing",
        )
    if not math.isfinite(delta):
        return PromotionDecision(
            promotion_state=ROLLED_BACK,
            reason="heldout_delta_invalid",
            rollback_reason="heldout_delta_invalid",
        )
    if delta < 0.0:
        return PromotionDecision(
            promotion_state=ROLLED_BACK,
            reason="heldout_delta_negative",
            rollback_reason="heldout_delta_negative",
        )
    if delta == 0.0:
        return PromotionDecision(
            promotion_state=ROLLED_BACK,
            reason="heldout_delta_null",
            rollback_reason="heldout_delta_null",
        )
    if delta < float(promotion_threshold):
        return PromotionDecision(
            promotion_state=HELD,
            reason="heldout_delta_below_promotion_threshold",
            rollback_reason=None,
        )
    return PromotionDecision(
        promotion_state=PROMOTED,
        reason="heldout_delta_clears_promotion_threshold",
        rollback_reason=None,
    )


def make_memory_entry(
    *,
    failure_signature: str,
    candidate_predicate_or_set: Mapping[str, Any],
    provenance: Mapping[str, Any],
    deterministic_guard_result: bool | Mapping[str, Any],
    heldout_delta: float | int | Mapping[str, Any] | None,
    source_artifacts: Iterable[str],
    promotion_threshold: float = DEFAULT_PROMOTION_THRESHOLD,
) -> dict[str, Any]:
    """Build one schema-valid verifier-memory entry.

    The memory id is derived only from the failure signature and candidate
    definition, so re-ingesting the same upstream evidence is idempotent.
    """
    delta_record = _heldout_delta_record(heldout_delta, promotion_threshold)
    decision = decide_promotion(
        deterministic_guard_result=deterministic_guard_result,
        heldout_delta=delta_record,
        promotion_threshold=promotion_threshold,
    )
    entry = {
        "memory_id": _memory_id(failure_signature, candidate_predicate_or_set),
        "failure_signature": str(failure_signature),
        "candidate_predicate_or_set": dict(candidate_predicate_or_set),
        "provenance": dict(provenance),
        "deterministic_guard_result": _guard_record(deterministic_guard_result),
        "heldout_delta": delta_record,
        "promotion_state": decision.promotion_state,
        "promotion_reason": decision.reason,
        "rollback_reason": decision.rollback_reason,
        "source_artifacts": sorted({str(path) for path in source_artifacts}),
    }
    assert_no_test_gold_leak(entry)
    return entry


def dedupe_memory_entries(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse duplicate memory entries by `memory_id`.

    Source artifacts are unioned so repeated ingestion preserves provenance
    without multiplying memory rows.
    """
    merged: dict[str, dict[str, Any]] = {}
    for raw_entry in entries:
        entry = dict(raw_entry)
        memory_id = str(entry.get("memory_id") or _memory_id_from_entry(entry))
        entry["memory_id"] = memory_id
        entry["source_artifacts"] = sorted(
            {str(path) for path in entry.get("source_artifacts", [])}
        )
        if memory_id not in merged:
            merged[memory_id] = entry
            continue
        prior = merged[memory_id]
        prior["source_artifacts"] = sorted(
            set(prior.get("source_artifacts", [])) | set(entry["source_artifacts"])
        )
    result = [merged[key] for key in sorted(merged)]
    assert_no_test_gold_leak({"entries": result})
    return result


def assert_no_test_gold_leak(
    payload: Any,
    *,
    forbidden_tokens: Iterable[str] = FORBIDDEN_TEST_GOLD_TOKENS,
) -> bool:
    """Reject memory payloads that carry test-gold or oracle-answer material."""
    tokens = {str(token) for token in forbidden_tokens}
    _check_no_forbidden_tokens(payload, tokens, path="$")
    return True


def _guard_passed(result: bool | Mapping[str, Any]) -> bool:
    if isinstance(result, bool):
        return result
    if not isinstance(result, Mapping):
        return False
    if result.get("passed") is False:
        return False
    if result.get("no_test_gold_leak") is False:
        return False
    if result.get("leakage_audit_passed") is False:
        return False
    checks = result.get("checks", {})
    if isinstance(checks, Mapping):
        return all(value is not False for value in checks.values())
    return bool(result.get("passed", True))


def _heldout_delta_value(heldout_delta: float | int | Mapping[str, Any] | None) -> float | None:
    raw_value = heldout_delta.get("value") if isinstance(heldout_delta, Mapping) else heldout_delta
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return math.nan


def _heldout_delta_record(
    heldout_delta: float | int | Mapping[str, Any] | None,
    promotion_threshold: float,
) -> dict[str, Any]:
    if isinstance(heldout_delta, Mapping):
        record = dict(heldout_delta)
    else:
        record = {"metric": "heldout_delta", "value": heldout_delta}
    record.setdefault("promotion_threshold", float(promotion_threshold))
    return record


def _guard_record(result: bool | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, Mapping):
        record = dict(result)
    else:
        record = {"passed": bool(result), "checks": {}}
    record.setdefault("passed", _guard_passed(result))
    record.setdefault("checks", {})
    return record


def _memory_id(
    failure_signature: str,
    candidate_predicate_or_set: Mapping[str, Any],
) -> str:
    digest = sha256(
        _canonical_json(
            {
                "failure_signature": str(failure_signature),
                "candidate_predicate_or_set": dict(candidate_predicate_or_set),
            }
        ).encode("utf-8")
    ).hexdigest()
    return f"verifier-memory:{digest[:16]}"


def _memory_id_from_entry(entry: Mapping[str, Any]) -> str:
    return _memory_id(
        str(entry.get("failure_signature", "")),
        _mapping_or_empty(entry.get("candidate_predicate_or_set")),
    )


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _check_no_forbidden_tokens(value: Any, tokens: set[str], *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in tokens:
                raise ValueError(f"forbidden test-gold token at {path}.{key_text}: {key_text}")
            _check_no_forbidden_tokens(item, tokens, path=f"{path}.{key_text}")
        return
    if isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            _check_no_forbidden_tokens(item, tokens, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        for token in tokens:
            if token in value:
                raise ValueError(f"forbidden test-gold token at {path}: {token}")
