"""Rebuild obligation authority and replay both arbitration arms.

This module reads the frozen scenario contract directly. It does not accept
producer labels as inputs. Exact hard and binding checks therefore remain the
authority for selection, while soft progress only breaks safe ties.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping, Sequence


ARBITER_VERSION = "carnot.exp6824.lexicographic_arbiter.v1"
MODULE_PATH = "python/carnot/experiment_6824_cold_arbiter.py"
SELECTIVE_ARM = "selective_priority"
FLAT_ARM = "flat_reject_retry"
ARMS = (SELECTIVE_ARM, FLAT_ARM)
PRIORITY_RANK = {"hard": 0, "binding": 1, "soft": 2}
CONTRACT_FIELDS = {
    "authority",
    "execution_consequence",
    "fallback",
    "prerequisite",
    "priority",
}


class ColdArbiterError(ValueError):
    """Report an incomplete frozen obligation contract."""


def canonical_action_bytes(action: Mapping[str, Any]) -> bytes:
    """Encode an action exactly as the producer's frozen JSON contract requires."""

    return json.dumps(
        action,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def action_sha256(action_bytes: bytes) -> str:
    """Bind the selected action identity to its exact bytes."""

    return "sha256:" + hashlib.sha256(action_bytes).hexdigest()


def rebuild_obligations(scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate all five contract fields and freeze authority-first order."""

    source = scenario.get("obligations")
    if not isinstance(source, list) or not source:
        raise ColdArbiterError("obligations must be a non-empty list")
    rebuilt: list[dict[str, Any]] = []
    for row in source:
        if not isinstance(row, dict) or set(row) != {"action", "contract", "obligation_id"}:
            raise ColdArbiterError("obligation record fields are incomplete")
        contract = row["contract"]
        if not isinstance(contract, dict):
            raise ColdArbiterError("contract must be an object")
        missing = sorted(CONTRACT_FIELDS.difference(contract))
        if missing:
            raise ColdArbiterError(f"missing contract field: {missing[0]}")
        authority = contract["authority"]
        prerequisite = contract["prerequisite"]
        priority = contract["priority"]
        fallback = contract["fallback"]
        consequence = contract["execution_consequence"]
        if not isinstance(authority, dict) or set(authority) != {"issuer", "order"}:
            raise ColdArbiterError("authority contract is invalid")
        if not isinstance(prerequisite, dict) or set(prerequisite) != {"all_of", "none_of"}:
            raise ColdArbiterError("prerequisite contract is invalid")
        if not isinstance(priority, dict) or set(priority) != {"class", "weight"}:
            raise ColdArbiterError("priority contract is invalid")
        if priority["class"] not in PRIORITY_RANK:
            raise ColdArbiterError("priority class is invalid")
        if not isinstance(fallback, dict) or set(fallback) != {"action", "reason"}:
            raise ColdArbiterError("fallback contract is invalid")
        if not isinstance(consequence, dict) or set(consequence) != {"add", "remove"}:
            raise ColdArbiterError("execution consequence is invalid")
        rebuilt.append(
            {
                "action": deepcopy(row["action"]),
                "authority_order": int(authority["order"]),
                "consequence": deepcopy(consequence),
                "fallback_action": deepcopy(fallback["action"]),
                "issuer": str(authority["issuer"]),
                "obligation_id": str(row["obligation_id"]),
                "prerequisite": deepcopy(prerequisite),
                "priority_class": str(priority["class"]),
                "priority_weight": int(priority["weight"]),
            }
        )
    return sorted(
        rebuilt,
        key=lambda row: (
            PRIORITY_RANK[row["priority_class"]],
            row["authority_order"],
            row["obligation_id"],
        ),
    )


def _active(obligation: Mapping[str, Any], facts: set[str]) -> bool:
    """Apply both positive and negative frozen prerequisites."""

    prerequisite = obligation["prerequisite"]
    return set(prerequisite["all_of"]).issubset(facts) and facts.isdisjoint(prerequisite["none_of"])


def _supported(obligation: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    """Require both the exact action and its declared authority issuer."""

    return (
        candidate["action"] == obligation["action"]
        and obligation["issuer"] in candidate["authority_chain"]
    )


def evaluate_candidate(scenario: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Compute proposal-time evidence from the frozen contract only."""

    base = deepcopy(dict(candidate))
    if candidate.get("parse_state") != "complete":
        return {
            **base,
            "authority_preserved": False,
            "binding_violation_vector": [],
            "first_conflict": "response_schema",
            "first_conflict_priority": "syntax",
            "hard_violation_count": 0,
            "legal_support": False,
            "soft_score": None,
            "transition": {"add": [], "remove": []},
        }
    obligations = rebuild_obligations(scenario)
    facts = set(scenario.get("observed_facts", []))
    active = [row for row in obligations if _active(row, facts)]
    hard = [row for row in active if row["priority_class"] == "hard"]
    binding = [row for row in active if row["priority_class"] == "binding"]
    hard_failures = [row for row in hard if not _supported(row, candidate)]
    binding_vector = [0 if _supported(row, candidate) else 1 for row in binding]
    matching = [row for row in active if _supported(row, candidate)]
    matching_action = [row for row in active if row["action"] == candidate["action"]]
    authority_preserved = all(
        row["issuer"] in candidate["authority_chain"] for row in matching_action
    )
    first_conflict: str | None = None
    first_priority: str | None = None
    if hard_failures:
        first_conflict = hard_failures[0]["obligation_id"]
        first_priority = "hard"
    elif 1 in binding_vector:
        index = binding_vector.index(1)
        first_conflict = binding[index]["obligation_id"]
        first_priority = "binding"
    elif not matching:
        first_conflict = "unbound_candidate"
        first_priority = "support"
    legal = not hard_failures and not any(binding_vector) and bool(matching)
    add = sorted({item for row in matching for item in row["consequence"]["add"]})
    remove = sorted({item for row in matching for item in row["consequence"]["remove"]})
    soft_bonus = sum(row["priority_weight"] for row in matching if row["priority_class"] == "soft")
    return {
        **base,
        "authority_preserved": authority_preserved,
        "binding_violation_vector": binding_vector,
        "first_conflict": first_conflict,
        "first_conflict_priority": first_priority,
        "hard_violation_count": len(hard_failures),
        "legal_support": legal,
        "soft_score": candidate["soft_progress"] + soft_bonus,
        "transition": {"add": add, "remove": remove},
    }


def _rejection(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    """Name the first higher-priority reason an invalid candidate loses."""

    if candidate.get("parse_state") != "complete":
        return {
            "candidate_id": candidate["candidate_id"],
            "first_conflict": "response_schema",
            "priority": "syntax",
        }
    if candidate.get("hard_violation_count", 0):
        return {
            "candidate_id": candidate["candidate_id"],
            "first_conflict": candidate["first_conflict"],
            "priority": "hard",
        }
    if any(candidate.get("binding_violation_vector", [])):
        return {
            "candidate_id": candidate["candidate_id"],
            "first_conflict": candidate["first_conflict"],
            "priority": "binding",
        }
    if not candidate.get("legal_support"):
        return {
            "candidate_id": candidate["candidate_id"],
            "first_conflict": "unbound_candidate",
            "priority": "support",
        }
    return None


def _post_hard_count(scenario: Mapping[str, Any], action: Mapping[str, Any]) -> int:
    """Count active hard actions that the selected transition does not satisfy."""

    if not scenario.get("obligations"):
        return 0
    facts = set(scenario.get("observed_facts", []))
    return sum(
        row["priority_class"] == "hard" and _active(row, facts) and row["action"] != action
        for row in rebuild_obligations(scenario)
    )


def _no_active_obligation(scenario: Mapping[str, Any]) -> bool:
    """Identify a declared fallback caused by stale prerequisites."""

    facts = set(scenario.get("observed_facts", []))
    return not any(_active(row, facts) for row in rebuild_obligations(scenario))


def valid_base_action(
    scenario: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    """Return the frozen safe proposal or a fully valid candidate-zero action."""

    safe = scenario.get("safe_proposal")
    if isinstance(safe, dict):
        return safe
    base = next((row for row in candidates if row.get("candidate_index") == 0), None)
    if (
        base is not None
        and base.get("parse_state") == "complete"
        and base.get("hard_violation_count") == 0
        and not any(base.get("binding_violation_vector", []))
        and base.get("authority_preserved")
        and base.get("legal_support")
    ):
        return base["action"]
    return None


def select_arm(
    scenario: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    soft_clip: int,
) -> dict[str, Any]:
    """Replay one arm while preserving the frozen proposal-time boundary."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    safe = valid_base_action(scenario, candidates)
    selected: Mapping[str, Any] | None = None
    selected_id: str | None = None
    if arm == SELECTIVE_ARM and isinstance(safe, dict):
        action = deepcopy(safe)
        selected_id = "base_proposal"
        certificate = {
            "first_higher_priority_conflict": None,
            "kind": "no_op_preserved",
            "rejections": [],
        }
    else:
        eligible = [
            row
            for row in candidates
            if row.get("hard_violation_count") == 0 and row.get("legal_support")
        ]
        if arm == SELECTIVE_ARM and eligible:
            selected = min(
                eligible,
                key=lambda row: (
                    tuple(row.get("binding_violation_vector", [])),
                    -max(-soft_clip, min(soft_clip, int(row.get("soft_score", 0)))),
                    int(row["candidate_index"]),
                ),
            )
        elif arm == FLAT_ARM:
            selected = next(
                (
                    row
                    for row in sorted(candidates, key=lambda item: int(item["candidate_index"]))
                    if row.get("parse_state") == "complete"
                    and row.get("hard_violation_count") == 0
                    and not any(row.get("binding_violation_vector", []))
                    and row.get("authority_preserved")
                    and row.get("legal_support")
                ),
                None,
            )
        if selected is not None:
            action = deepcopy(selected["action"])
            selected_id = str(selected["candidate_id"])
        else:
            action = deepcopy(scenario["fallback_action"])
        rejections = [item for row in candidates if (item := _rejection(row)) is not None]
        certificate = {
            "first_higher_priority_conflict": (
                rejections[0]["first_conflict"] if rejections else None
            ),
            "kind": "selected" if selected is not None else "no_legal_candidate",
            "rejections": rejections,
        }
    action_bytes = canonical_action_bytes(action)
    safe_bytes = canonical_action_bytes(safe) if isinstance(safe, dict) else None
    post_hard = _post_hard_count(scenario, action)
    abstention = selected_id is None
    return {
        "abstention": abstention,
        "accepted_hard_violation": not abstention and post_hard > 0,
        "accepted_progress": 0 if abstention else 1,
        "certificate": certificate,
        "certificate_complete": True,
        "false_intervention": safe_bytes is not None and action_bytes != safe_bytes,
        "harmful_selection": not abstention and post_hard > 0,
        "legality": (selected_id is not None and post_hard == 0)
        or action == scenario["fallback_action"],
        "outcome_certificate": {
            "kind": "no_candidate" if abstention else "selected",
            "reason": (
                "stale_prerequisite"
                if abstention and _no_active_obligation(scenario)
                else "no_legal_candidate"
                if abstention
                else "exact_lexicographic_minimum"
            ),
        },
        "post_selection_hard_violation_count": post_hard,
        "retry_count": 1 if abstention else 0,
        "safe_action_identity": action_bytes == safe_bytes if safe_bytes is not None else None,
        "selected_action": action,
        "selected_action_bytes": action_bytes,
        "selected_action_bytes_b64": base64.b64encode(action_bytes).decode("ascii"),
        "selected_action_sha256": action_sha256(action_bytes),
        "selected_candidate_id": selected_id,
    }
