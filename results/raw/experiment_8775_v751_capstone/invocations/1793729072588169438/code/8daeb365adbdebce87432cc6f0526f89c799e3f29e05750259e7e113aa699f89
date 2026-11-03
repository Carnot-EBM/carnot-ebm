"""Exact action-to-outcome transport for the canonical live E3 path.

REQ-ARC-WMTE-6681 requires explicit identifiers at every boundary. This
module records policy proposals, adapter applications, environment calls, and
environment returns as separate events. The reducer joins those events by
their identifiers, so event order and clock precision cannot change lineage.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
from enum import Enum
import hashlib
import json
from typing import Any


JsonDict = dict[str, Any]
EVENT_KEYS = ("proposals", "applications", "environment_steps", "outcomes")


class OutcomeLineageError(ValueError):
    """Raised when an identity is duplicated or a live boundary is ambiguous."""


def canonical_json(value: Any) -> str:
    """Return stable JSON for identities and hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    """Return a labeled SHA-256 over canonical JSON."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    """Convert SDK and NumPy values without changing their data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json"))
    return str(value)


def normalize_action(action: Any) -> JsonDict:
    """Normalize a policy tuple, event dict, or SDK GameAction."""

    if isinstance(action, Mapping):
        kind = action.get("kind", action.get("action"))
        data = action.get("data")
    elif isinstance(action, tuple) and len(action) == 2:
        kind, data = action
    else:
        name = str(getattr(action, "name", "") or "")
        kind = "RESET" if name == "RESET" else getattr(action, "value", action)
        action_data = getattr(action, "action_data", None)
        data = action_data.model_dump() if hasattr(action_data, "model_dump") else None
    if kind is None:
        kind = "RESET"
    if isinstance(kind, str):
        if kind.upper() == "RESET":
            kind = "RESET"
        elif kind.upper().startswith("ACTION") and kind[6:].isdigit():
            kind = int(kind[6:])
    cleaned = _json_value(data)
    if isinstance(cleaned, dict):
        cleaned = {key: value for key, value in cleaned.items() if key != "game_id"}
        cleaned = cleaned or None
    return {"kind": kind, "data": cleaned}


def normalize_observation(observation: Any) -> JsonDict | None:
    """Copy the complete public ARC observation returned at the step seam."""

    if observation is None:
        return None
    if isinstance(observation, Mapping):
        return _json_value(observation)
    fields = (
        "game_id",
        "frame",
        "state",
        "levels_completed",
        "win_levels",
        "action_input",
        "guid",
        "full_reset",
        "available_actions",
    )
    return {field: _json_value(getattr(observation, field, None)) for field in fields}


def observation_hash(observation: Any) -> str:
    """Hash the exact normalized observation."""

    return sha256_json(normalize_observation(observation))


def _unpack_environment_return(returned: Any) -> tuple[Any, JsonDict, JsonDict, str]:
    """Preserve Gym rewards when present and ARC's explicit absence otherwise."""

    reward: JsonDict
    termination: JsonDict
    if isinstance(returned, tuple) and len(returned) == 5:
        observation, reward_value, terminated, truncated, _info = returned
        reward = {
            "present": True,
            "value": _json_value(reward_value),
            "source": "environment_step_return[1]",
            "synthetic": False,
        }
        termination = {
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "state": _json_value(getattr(observation, "state", None)),
            "source": "environment_step_return[2:4]",
        }
        return observation, reward, termination, "tuple5"
    if isinstance(returned, tuple) and len(returned) in (3, 4):
        observation, reward_value, done = returned[:3]
        reward = {
            "present": True,
            "value": _json_value(reward_value),
            "source": "environment_step_return[1]",
            "synthetic": False,
        }
        termination = {
            "terminated": bool(done),
            "truncated": False,
            "state": _json_value(getattr(observation, "state", None)),
            "source": "environment_step_return[2]",
        }
        return observation, reward, termination, f"tuple{len(returned)}"
    observation = returned
    state = _json_value(getattr(observation, "state", None))
    reward = {
        "present": False,
        "value": None,
        "source": "arc_agi.FrameDataRaw.step_return_schema",
        "synthetic": False,
    }
    termination = {
        "terminated": state in {"WIN", "GAME_OVER"},
        "truncated": False,
        "state": state,
        "source": "arc_agi.FrameDataRaw.state",
    }
    return observation, reward, termination, "arc_agi.FrameDataRaw"


class E3OutcomeTransport:
    """Record one episode's four live boundary events.

    The transport permits one in-flight action. This restriction is useful:
    the ARC agent loop is synchronous, so a second proposal before the first
    return means the measurement no longer knows which action caused a frame.
    """

    def __init__(
        self,
        *,
        family: str,
        attempt: int,
        episode_seed: int,
        episode_id: str,
    ) -> None:
        self.family = str(family)
        self.attempt = int(attempt)
        self.episode_seed = int(episode_seed)
        self.episode_id = str(episode_id)
        self._events: dict[str, list[JsonDict]] = {key: [] for key in EVENT_KEYS}
        self._pending: JsonDict | None = None

    def _identity(self, kind: str, payload: Mapping[str, Any]) -> str:
        return sha256_json(
            {
                "event_kind": kind,
                "episode_id": self.episode_id,
                "sequence": len(self._events[f"{kind}s"]),
                "payload": payload,
            }
        )

    def record_proposal(
        self,
        *,
        proposed_action: Any,
        policy_selected_action: Any,
        observation_before: Any,
        supervisor_decision: Mapping[str, Any] | None,
    ) -> str:
        """Record the policy decision before the adapter creates a GameAction."""

        if self._pending is not None:
            raise OutcomeLineageError("previous proposal has no terminal outcome")
        before = normalize_observation(observation_before)
        payload: JsonDict = {
            "episode_id": self.episode_id,
            "family": self.family,
            "family_role": "held",
            "attempt": self.attempt,
            "episode_seed": self.episode_seed,
            "action_index": len(self._events["proposals"]),
            "state_hash": sha256_json(before),
            "observation_before": before,
            "proposed_action": normalize_action(proposed_action),
            "policy_selected_action": normalize_action(policy_selected_action),
            "supervisor_decision": _json_value(dict(supervisor_decision or {})),
        }
        proposal_id = self._identity("proposal", payload)
        event = {"proposal_id": proposal_id, **payload}
        self._events["proposals"].append(event)
        self._pending = {"proposal": event}
        return proposal_id

    def record_application(self, applied_action: Any) -> str:
        """Bind the adapter's actual SDK action to the current proposal."""

        if self._pending is None or "application" in self._pending:
            raise OutcomeLineageError("application has no unique pending proposal")
        proposal = self._pending["proposal"]
        applied = normalize_action(applied_action)
        if applied != proposal["policy_selected_action"]:
            raise OutcomeLineageError("adapter action does not match policy-selected action")
        payload = {
            "proposal_id": proposal["proposal_id"],
            "applied_action": applied,
            "action_cost": 1,
        }
        application_id = self._identity("application", payload)
        event = {"application_id": application_id, **payload}
        self._events["applications"].append(event)
        self._pending["application"] = event
        return application_id

    def begin_environment_step(self, actual_action: Any) -> str:
        """Bind the action passed to the exact synchronous environment call."""

        if self._pending is None or "application" not in self._pending:
            raise OutcomeLineageError("environment step has no application")
        if "environment_step" in self._pending:
            raise OutcomeLineageError("application already has an environment step")
        application = self._pending["application"]
        actual = normalize_action(actual_action)
        if actual != application["applied_action"]:
            raise OutcomeLineageError("environment action does not match applied action")
        proposal = self._pending["proposal"]
        payload = {
            "application_id": application["application_id"],
            "actual_action": actual,
            "observation_before_hash": proposal["state_hash"],
            "live_environment": True,
        }
        step_id = self._identity("environment_step", payload)
        event = {"environment_step_id": step_id, **payload}
        self._events["environment_steps"].append(event)
        self._pending["environment_step"] = event
        return step_id

    def record_environment_return(self, environment_step_id: str, returned: Any) -> str:
        """Record the exact successful return and close the in-flight action."""

        self._require_step(environment_step_id)
        observation, reward, termination, schema = _unpack_environment_return(returned)
        after = normalize_observation(observation)
        payload: JsonDict = {
            "environment_step_id": str(environment_step_id),
            "status": "returned",
            "observation_after": after,
            "observation_after_hash": sha256_json(after),
            "reward": reward,
            "termination": termination,
            "return_schema": schema,
            "return_hash": sha256_json(
                {"observation": after, "reward": reward, "termination": termination}
            ),
            "live_return": True,
            "error": None,
        }
        outcome_id = self._identity("outcome", payload)
        self._events["outcomes"].append({"outcome_id": outcome_id, **payload})
        self._pending = None
        return outcome_id

    def record_environment_failure(
        self, environment_step_id: str, *, status: str, error: str
    ) -> str:
        """Keep a timeout or environment error as an ineligible outcome row."""

        self._require_step(environment_step_id)
        if status not in {"timeout", "environment_error"}:
            raise ValueError("failure status must be timeout or environment_error")
        payload: JsonDict = {
            "environment_step_id": str(environment_step_id),
            "status": status,
            "observation_after": None,
            "observation_after_hash": sha256_json(None),
            "reward": {
                "present": False,
                "value": None,
                "source": "environment_step_failure",
                "synthetic": False,
            },
            "termination": {
                "terminated": None,
                "truncated": status == "timeout",
                "state": None,
                "source": "environment_step_failure",
            },
            "return_schema": "failure",
            "return_hash": sha256_json({"status": status, "error": str(error)}),
            "live_return": False,
            "error": str(error),
        }
        outcome_id = self._identity("outcome", payload)
        self._events["outcomes"].append({"outcome_id": outcome_id, **payload})
        self._pending = None
        return outcome_id

    def _require_step(self, environment_step_id: str) -> JsonDict:
        if self._pending is None or "environment_step" not in self._pending:
            raise OutcomeLineageError("outcome has no environment step")
        pending = self._pending["environment_step"]
        if pending["environment_step_id"] != environment_step_id:
            raise OutcomeLineageError("outcome environment_step_id mismatch")
        return pending

    def events(self) -> dict[str, list[JsonDict]]:
        """Return a defensive copy of all event rows."""

        return copy.deepcopy(self._events)


def _unique_index(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identity = str(row.get(key) or "")
        if not identity:
            raise OutcomeLineageError(f"missing {key}")
        if identity in indexed:
            raise OutcomeLineageError(f"duplicate {key}: {identity}")
        indexed[identity] = row
    return indexed


def _children(
    rows: Sequence[Mapping[str, Any]], parent_key: str
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(parent_key) or "")].append(row)
    return grouped


def join_outcome_events(
    events: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[JsonDict], JsonDict]:
    """Join event tables by explicit foreign keys and report every ambiguity."""

    tables = {key: list(events.get(key) or []) for key in EVENT_KEYS}
    proposals = _unique_index(tables["proposals"], "proposal_id")
    application_index = _unique_index(tables["applications"], "application_id")
    step_index = _unique_index(tables["environment_steps"], "environment_step_id")
    _unique_index(tables["outcomes"], "outcome_id")
    applications = _children(tables["applications"], "proposal_id")
    steps = _children(tables["environment_steps"], "application_id")
    outcomes = _children(tables["outcomes"], "environment_step_id")
    issues: list[JsonDict] = []
    joined: list[JsonDict] = []

    ordered_proposals = sorted(
        proposals.values(),
        key=lambda row: (
            str(row.get("episode_id") or ""),
            int(row.get("action_index") or 0),
            str(row.get("proposal_id") or ""),
        ),
    )
    for proposal in ordered_proposals:
        proposal_id = str(proposal["proposal_id"])
        proposal_issues: list[str] = []
        app_rows = applications.get(proposal_id, [])
        if len(app_rows) != 1:
            issues.append(
                {
                    "proposal_id": proposal_id,
                    "reason": "application_child_count",
                    "observed": len(app_rows),
                    "expected": 1,
                }
            )
            continue
        application = app_rows[0]
        step_rows = steps.get(str(application["application_id"]), [])
        if len(step_rows) != 1:
            issues.append(
                {
                    "proposal_id": proposal_id,
                    "application_id": application["application_id"],
                    "reason": "environment_step_child_count",
                    "observed": len(step_rows),
                    "expected": 1,
                }
            )
            continue
        step = step_rows[0]
        outcome_rows = outcomes.get(str(step["environment_step_id"]), [])
        if len(outcome_rows) != 1:
            issues.append(
                {
                    "proposal_id": proposal_id,
                    "environment_step_id": step["environment_step_id"],
                    "reason": "outcome_child_count",
                    "observed": len(outcome_rows),
                    "expected": 1,
                }
            )
            continue
        outcome = outcome_rows[0]
        if application.get("applied_action") != proposal.get("policy_selected_action"):
            proposal_issues.append("application_action_mismatch")
        if step.get("actual_action") != application.get("applied_action"):
            proposal_issues.append("environment_action_mismatch")
        if step.get("observation_before_hash") != proposal.get("state_hash"):
            proposal_issues.append("stale_observation")
        reward = outcome.get("reward") or {}
        if reward.get("synthetic") is not False or reward.get("source") not in {
            "arc_agi.FrameDataRaw.step_return_schema",
            "environment_step_return[1]",
            "environment_step_failure",
        }:
            proposal_issues.append("synthetic_reward")
        if outcome.get("status") != "returned" or outcome.get("live_return") is not True:
            proposal_issues.append(str(outcome.get("status") or "missing_live_return"))
        for reason in proposal_issues:
            issues.append({"proposal_id": proposal_id, "reason": reason})

        before = proposal.get("observation_before") or {}
        after = outcome.get("observation_after") or {}
        before_level = before.get("levels_completed")
        after_level = after.get("levels_completed")
        decision = proposal.get("supervisor_decision") or {}
        redirect = bool(decision.get("fired")) and (
            proposal.get("proposed_action") != proposal.get("policy_selected_action")
        )
        lineage = {
            "proposal_id": proposal_id,
            "application_id": application["application_id"],
            "environment_step_id": step["environment_step_id"],
            "outcome_id": outcome["outcome_id"],
        }
        joined.append(
            {
                **lineage,
                "lineage": lineage,
                "episode_id": proposal.get("episode_id"),
                "family": proposal.get("family"),
                "family_role": proposal.get("family_role"),
                "attempt": proposal.get("attempt"),
                "episode_seed": proposal.get("episode_seed"),
                "action_index": proposal.get("action_index"),
                "state_hash": proposal.get("state_hash"),
                "proposed_action": proposal.get("proposed_action"),
                "policy_selected_action": proposal.get("policy_selected_action"),
                "applied_action": application.get("applied_action"),
                "redirect_applied": redirect,
                "redirect_reason": decision.get("arm") if redirect else None,
                "supervisor_decision": decision,
                "observation_before": proposal.get("observation_before"),
                "observation_after": outcome.get("observation_after"),
                "reward": outcome.get("reward"),
                "termination": outcome.get("termination"),
                "level_change": bool(
                    before_level is not None
                    and after_level is not None
                    and int(after_level) != int(before_level)
                ),
                "levels_completed_before": before_level,
                "levels_completed_after": after_level,
                "action_cost": application.get("action_cost"),
                "outcome_status": outcome.get("status"),
                "return_schema": outcome.get("return_schema"),
                "return_hash": outcome.get("return_hash"),
                "live_return": outcome.get("live_return"),
                "error": outcome.get("error"),
                "fully_joined": not proposal_issues,
            }
        )

    parent_contracts = (
        ("application", tables["applications"], "proposal_id", proposals),
        (
            "environment_step",
            tables["environment_steps"],
            "application_id",
            application_index,
        ),
        ("outcome", tables["outcomes"], "environment_step_id", step_index),
    )
    for child_kind, child_rows, parent_key, parent_index in parent_contracts:
        for child in child_rows:
            parent_id = str(child.get(parent_key) or "")
            if parent_id not in parent_index:
                issues.append(
                    {
                        "reason": f"orphan_{child_kind}",
                        "parent_key": parent_key,
                        "parent_id": parent_id,
                    }
                )

    redirect_proposals = sum(
        int(
            bool((proposal.get("supervisor_decision") or {}).get("fired"))
            and proposal.get("proposed_action") != proposal.get("policy_selected_action")
        )
        for proposal in proposals.values()
    )
    eligible_redirects = sum(
        int(row["redirect_applied"] and row["fully_joined"] and row["family_role"] == "held")
        for row in joined
    )
    ready = bool(redirect_proposals > 0 and eligible_redirects == redirect_proposals and not issues)
    return joined, {
        "ready": ready,
        "proposal_count": len(proposals),
        "joined_count": len(joined),
        "redirect_proposal_count": redirect_proposals,
        "eligible_redirect_count": eligible_redirects,
        "issue_count": len(issues),
        "issues": issues,
    }


def run_lineage_attacks(events: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[JsonDict]:
    """Apply the required lineage mutations to one known-good event bundle."""

    base = {key: [dict(row) for row in events.get(key, [])] for key in EVENT_KEYS}
    expected_rows, _ = join_outcome_events(base)
    attacks: list[tuple[str, dict[str, list[JsonDict]], str]] = []

    duplicate = copy.deepcopy(base)
    duplicate["outcomes"].append(copy.deepcopy(duplicate["outcomes"][0]))
    attacks.append(("duplicated_ids", duplicate, "rejected"))
    missing = copy.deepcopy(base)
    missing["outcomes"].clear()
    attacks.append(("dropped_outcomes", missing, "rejected"))
    reordered = {key: list(reversed(value)) for key, value in copy.deepcopy(base).items()}
    attacks.append(("reordered_events", reordered, "joined_by_identity"))
    stale = copy.deepcopy(base)
    stale["environment_steps"][0]["observation_before_hash"] = "sha256:stale"
    attacks.append(("stale_observations", stale, "rejected"))
    mismatch = copy.deepcopy(base)
    mismatch["applications"][0]["applied_action"] = {"kind": 2, "data": None}
    attacks.append(("mismatched_actions", mismatch, "rejected"))
    synthetic = copy.deepcopy(base)
    synthetic["outcomes"][0]["reward"] = {
        "present": True,
        "value": 1,
        "source": "derived_level_change",
        "synthetic": True,
    }
    attacks.append(("synthetic_rewards", synthetic, "rejected"))
    timeout = copy.deepcopy(base)
    timeout["outcomes"][0].update(status="timeout", live_return=False)
    attacks.append(("timeout", timeout, "rejected"))
    environment_error = copy.deepcopy(base)
    environment_error["outcomes"][0].update(status="environment_error", live_return=False)
    attacks.append(("environment_error", environment_error, "rejected"))
    partial = copy.deepcopy(base)
    partial["environment_steps"].clear()
    attacks.append(("partial_writes", partial, "rejected"))

    rows: list[JsonDict] = []
    for attack_id, candidate, expected in attacks:
        try:
            joined, audit = join_outcome_events(candidate)
            observed = (
                "joined_by_identity"
                if attack_id == "reordered_events" and joined == expected_rows and audit["ready"]
                else ("accepted" if audit["ready"] else "rejected")
            )
            issues = audit["issues"]
        except OutcomeLineageError as exc:
            observed = "rejected"
            issues = [{"reason": str(exc)}]
        rows.append(
            {
                "attack_id": attack_id,
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
                "issues": issues,
            }
        )
    return rows
