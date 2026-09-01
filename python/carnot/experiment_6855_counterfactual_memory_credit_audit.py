"""Audit causal credit for observed bounded controller-state writes.

The source controller stores only small sufficient statistics. This module
reimplements that reducer so the audit does not trust the producer's summary.
It changes state transitions only when an exact source receipt supports them.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_RELATIVE_PATH = Path(
    "results/experiment_6854_risk_sensitive_abstention_memory_controller.json"
)
FIXTURE_RELATIVE_PATH = Path(
    "results/experiment_6853_risk_sensitive_memory_opportunity_fixture.json"
)
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6855_counterfactual_memory_credit_audit.json"
)
SCHEMA = "carnot.experiment_6855.counterfactual_memory_credit_audit.v1"
EXPERIMENT_ID = "exp6855-counterfactual-memory-credit-audit"
INFERENCE_SUBSTRATE = "deterministic CPU counterfactual replay"
BLOCKED_VERDICT = "complete_blocked_counterfactual_memory_credit_audit"
RANDOM_SEED = 6_855_001
EXACT_COALITION_LIMIT = 8
COALITION_WINDOW = 9
PERMUTATION_COUNT = 8
STATE_SCHEMA = "carnot.risk_sensitive_contextual_bandit_state.v1"

ACTIONS = ("verified_memory", "no_memory", "abstain")
TIE_BREAK_ORDER = ("abstain", "no_memory", "verified_memory")
FAMILIES = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
CORRECTION_STATUSES = (
    "stale_correction_pending",
    "conflicting_correction_pending",
    "delayed_correction_pending",
    "no_correction_needed",
)
FEATURE_NAMES = (
    "bias",
    "relevance",
    "uncertainty",
    "exact_compatibility",
    "age_capped",
    "capacity_budget_capped",
    "factual_read_admitted",
    "false_positive_risk",
    *(f"family:{family}" for family in FAMILIES),
    *(f"correction:{status}" for status in CORRECTION_STATUSES),
)
PRIORS = {"verified_memory": 0.72, "no_memory": 0.30, "abstain": 0.25}
CONFIG = {
    "risk_ratio": 3.0,
    "ridge": 1.0,
    "confidence_scale": 0.12,
    "max_updates_per_action": 256,
    "max_pending": 4,
    "max_state_bytes": 16_384,
}

VALID_COUNTERFACTUAL_CONTRACT: JsonDict = {
    "declared_valid": True,
    "write_unit": "one observed Exp6854 bounded external-policy update receipt",
    "fixed_exogenous_inputs": [
        "decision identities",
        "chronological decision order",
        "pre-outcome contexts except the declared placebo feature",
        "action availability",
        "exact later outcome identities and signed directions",
    ],
    "deletion_support": (
        "Omit one observed sufficient-statistic transition and reconstruct every "
        "later selection state with the fresh reducer."
    ),
    "substitution_support": (
        "Use one different observed donor with the same action and feature vector."
    ),
    "order_support": (
        "Swap only observed writes with one shared feedback reveal boundary."
    ),
    "coalition_support": (
        "Use each eligible prior write at most once inside the bounded target window."
    ),
    "outcome_rule": (
        "The exact signed direction stays fixed. The declared risk matrix maps a "
        "selected action to loss; no environmental outcome is inferred."
    ),
    "state_divergence_rule": (
        "Reject nonfinite state, an out-of-bound update, or a write that does not "
        "precede the target boundary."
    ),
    "exact_coalition_limit": EXACT_COALITION_LIMIT,
    "bounded_window_size": COALITION_WINDOW,
    "larger_window_method": "seeded bounded permutations with 95% intervals",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "random_seed",
    "reproducibility_checksum",
    "rows",
    "valid_counterfactual_contract",
    "deletion_rows",
    "substitution_rows",
    "coalition_rows",
    "approximation_receipts",
    "unsupported_counterfactual_rows",
    "placebo_control_rows",
    "per_write_credit_summary",
    "harmful_write_count",
    "redundant_write_count",
    "interaction_witnesses",
    "selection_skill_effect",
    "stream_luck_effect",
    "counterfactual_memory_audit_complete_score",
    "causal_memory_credit_eligible_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

OPEN_SPEC_IDS = (
    "REQ-CL-6855",
    "SCENARIO-CL-6855-PRECONDITIONS",
    "SCENARIO-CL-6855-INVALID-COALITION",
    "SCENARIO-CL-6855-MISSING-SUPPORT",
    "SCENARIO-CL-6855-STATE-PATH-DIVERGENCE",
    "SCENARIO-CL-6855-DUPLICATE-WRITE",
    "SCENARIO-CL-6855-INTERACTION",
    "SCENARIO-CL-6855-PLACEBO",
    "SCENARIO-CL-6855-ZERO-HEADROOM",
    "SCENARIO-CL-6855-METHODS",
    "SCENARIO-CL-6855-CONTROLS",
    "SCENARIO-CL-6855-GATES",
)


class CounterfactualValidityError(ValueError):
    """Signal a transition that the declared support contract does not permit."""


@dataclass(frozen=True)
class WriteRecord:
    """Store one exact sufficient-statistic transition from a source receipt."""

    write_id: str
    decision_id: str
    update_sequence_index: int
    action: str
    features: tuple[float, ...]
    bounded_loss: float
    support_sha256: str


@dataclass(frozen=True)
class ReplaySelection:
    """Return a reconstructed state and its target action without an outcome guess."""

    state_sha256: str
    applied_write_ids: tuple[str, ...]
    selected_action: str
    pessimistic_scores: Mapping[str, float]


def canonical_json_bytes(value: Any) -> bytes:
    """Use stable bytes so hashes do not depend on JSON formatting."""

    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Hash structured evidence through the stable JSON representation."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Hash an input artifact or return no digest for an unreadable path."""

    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def load_source(path: Path) -> JsonDict:
    """Load one source object and turn file failures into blocked evidence."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"_load_error": type(error).__name__}
    if not isinstance(value, dict):
        return {"_load_error": "not_object"}
    return value


def source_artifact_hashes(repo_root: Path) -> JsonDict:
    """Bind the audit to the exact controller and opportunity artifacts."""

    return {
        "exp6854": {
            "path": str(CONTROLLER_RELATIVE_PATH),
            "sha256": sha256_file(repo_root / CONTROLLER_RELATIVE_PATH),
            "role": "completed controller decisions and bounded update receipts",
        },
        "exp6853": {
            "path": str(FIXTURE_RELATIVE_PATH),
            "sha256": sha256_file(repo_root / FIXTURE_RELATIVE_PATH),
            "role": "fixed contexts, action availability, and exact later outcomes",
        },
    }


def context_features(context: Mapping[str, Any]) -> tuple[float, ...]:
    """Encode the producer's fixed public schema without reading outcome fields."""

    capacity = context.get("capacity", {})
    if not isinstance(capacity, Mapping):
        raise CounterfactualValidityError("context capacity is invalid")
    family = str(context.get("family", ""))
    status = str(context.get("correction_status", ""))
    if family not in FAMILIES or status not in CORRECTION_STATUSES:
        raise CounterfactualValidityError("context category is unsupported")

    numbers = (
        context.get("relevance"),
        context.get("uncertainty"),
        context.get("age"),
        context.get("false_positive_risk"),
        capacity.get("budget"),
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in numbers
    ):
        raise CounterfactualValidityError("context number is unsupported")
    return (
        1.0,
        float(context["relevance"]),
        float(context["uncertainty"]),
        float(context.get("exact_compatibility") is True),
        min(float(context["age"]), 8.0) / 8.0,
        min(float(capacity["budget"]), 4.0) / 4.0,
        float(capacity.get("factual_read_admitted") is True),
        float(context["false_positive_risk"]),
        *(float(family == known) for known in FAMILIES),
        *(float(status == known) for known in CORRECTION_STATUSES),
    )


class FreshReducer:
    """Rebuild the diagonal policy statistics without producer aggregate code."""

    def __init__(self) -> None:
        """Create only fixed-size prior statistics."""

        self.stats: JsonDict = {
            action: {
                "precision": [CONFIG["ridge"]] * len(FEATURE_NAMES),
                "loss_sum": [CONFIG["ridge"] * PRIORS[action]] * len(FEATURE_NAMES),
                "updates": 0,
            }
            for action in ACTIONS
        }

    def copy(self) -> FreshReducer:
        """Copy the bounded statistics for one independent coalition branch."""

        clone = FreshReducer()
        clone.stats = {
            action: {
                "precision": list(row["precision"]),
                "loss_sum": list(row["loss_sum"]),
                "updates": int(row["updates"]),
            }
            for action, row in self.stats.items()
        }
        return clone

    def state_hash(self) -> str:
        """Hash the selection-relevant statistics used by counterfactual replay."""

        return sha256_json({"feature_names": list(FEATURE_NAMES), "stats": self.stats})

    def apply(self, write: WriteRecord) -> None:
        """Apply one supported bounded write or reject a divergent state path."""

        if write.action not in ACTIONS:
            raise CounterfactualValidityError("write action is unsupported")
        if len(write.features) != len(FEATURE_NAMES):
            raise CounterfactualValidityError("write feature length is unsupported")
        values = (*write.features, write.bounded_loss)
        if any(not math.isfinite(float(value)) for value in values):
            raise CounterfactualValidityError("write values must be finite")
        if not 0.0 <= write.bounded_loss <= CONFIG["risk_ratio"]:
            raise CounterfactualValidityError("write loss exceeds bounds")

        row = self.stats[write.action]
        if row["updates"] >= CONFIG["max_updates_per_action"]:
            decay = (CONFIG["max_updates_per_action"] - 1) / CONFIG[
                "max_updates_per_action"
            ]
            prior = PRIORS[write.action]
            row["precision"] = [
                CONFIG["ridge"] + (float(value) - CONFIG["ridge"]) * decay
                for value in row["precision"]
            ]
            row["loss_sum"] = [
                CONFIG["ridge"] * prior
                + (float(value) - CONFIG["ridge"] * prior) * decay
                for value in row["loss_sum"]
            ]
        row["precision"] = [
            float(value) + feature * feature
            for value, feature in zip(row["precision"], write.features, strict=True)
        ]
        row["loss_sum"] = [
            float(value) + feature * feature * write.bounded_loss
            for value, feature in zip(row["loss_sum"], write.features, strict=True)
        ]
        row["updates"] = min(
            CONFIG["max_updates_per_action"], int(row["updates"]) + 1
        )

    def scores(self, features: Sequence[float]) -> dict[str, float]:
        """Compute the declared pessimistic loss from reconstructed statistics."""

        if len(features) != len(FEATURE_NAMES):
            raise CounterfactualValidityError("target feature length is unsupported")
        weights = [abs(float(value)) for value in features]
        denominator = sum(weights) or 1.0
        output: dict[str, float] = {}
        for action in ACTIONS:
            row = self.stats[action]
            estimates = [
                float(total) / float(precision)
                for total, precision in zip(
                    row["loss_sum"], row["precision"], strict=True
                )
            ]
            mean = (
                sum(
                    weight * estimate
                    for weight, estimate in zip(weights, estimates, strict=True)
                )
                / denominator
            )
            uncertainty = (
                sum(
                    weight / math.sqrt(float(precision))
                    for weight, precision in zip(
                        weights, row["precision"], strict=True
                    )
                )
                / denominator
            )
            scale = {"verified_memory": 3.0, "no_memory": 1.0, "abstain": 0.6}[
                action
            ]
            output[action] = round(
                mean + CONFIG["confidence_scale"] * uncertainty * scale, 12
            )
        return output

    def select(self, features: Sequence[float]) -> tuple[str, dict[str, float]]:
        """Select the smallest pessimistic loss with the fixed safety tie order."""

        scores = self.scores(features)
        action = min(
            TIE_BREAK_ORDER,
            key=lambda item: (scores[item], TIE_BREAK_ORDER.index(item)),
        )
        return action, scores

    def replay(
        self,
        writes: Sequence[WriteRecord],
        *,
        target_boundary: int,
        target_features: Sequence[float] | None = None,
    ) -> ReplaySelection:
        """Reach one target boundary only through preceding supported writes."""

        identifiers = [write.write_id for write in writes]
        if len(identifiers) != len(set(identifiers)):
            raise CounterfactualValidityError("duplicate write in replay path")
        if any(write.update_sequence_index >= target_boundary for write in writes):
            raise CounterfactualValidityError("write does not precede target boundary")
        for write in sorted(
            writes, key=lambda item: (item.update_sequence_index, item.write_id)
        ):
            self.apply(write)
        features = target_features or ((1.0,) + (0.0,) * 14)
        action, scores = self.select(features)
        return ReplaySelection(
            state_sha256=self.state_hash(),
            applied_write_ids=tuple(identifiers),
            selected_action=action,
            pessimistic_scores=scores,
        )


def _source_state_hash(stats: Mapping[str, Any], pending: Mapping[str, Any]) -> str:
    """Reproduce the producer's complete state hash, including pending decisions."""

    payload = {
        "schema": STATE_SCHEMA,
        "config": CONFIG,
        "feature_names": list(FEATURE_NAMES),
        "action_stats": stats,
        "pending": {key: pending[key] for key in sorted(pending)},
    }
    return sha256_json(payload)


def _loss(action: str, direction: int) -> float:
    """Apply the declared risk matrix to one fixed exact direction."""

    if direction > 0:
        return {"verified_memory": 0.0, "no_memory": 1.0, "abstain": 0.6}[action]
    if direction < 0:
        return {"verified_memory": 3.0, "no_memory": 0.0, "abstain": 0.1}[action]
    return {"verified_memory": 0.25, "no_memory": 0.0, "abstain": 0.1}[action]


def _source_replay(controller: Mapping[str, Any], fixture: Mapping[str, Any]) -> JsonDict:
    """Recreate every source action and update receipt from fixed fixture inputs."""

    counters = {
        "decision_hash_mismatch_count": 0,
        "state_hash_mismatch_count": 0,
        "update_hash_mismatch_count": 0,
        "exact_outcome_mismatch_count": 0,
    }
    try:
        fixture_rows = list(fixture["rows"])
        controller_rows = {
            str(row["decision_id"]): row for row in controller["rows"]
        }
        action_sources = {
            str(row["decision_id"]): row
            for row in controller["pre_outcome_action_receipts"]
        }
        feedback_sources = {
            str(row["decision_id"]): row
            for row in controller["exact_feedback_receipts"]
        }
        update_sources = {
            str(row["decision_id"]): row for row in controller["update_rows"]
        }
        outcomes = {
            str(row["decision_id"]): row["exact_later_outcome"]
            for row in fixture_rows
        }
        contexts = {
            str(row["decision_id"]): row["decision_context"] for row in fixture_rows
        }
        delayed = {
            str(row["decision_id"]): bool(row.get("delayed_correction"))
            for row in fixture_rows
        }
        sequence = {
            str(row["decision_id"]): int(row["decision_sequence_index"])
            for row in fixture_rows
        }
        reducer = FreshReducer()
        pending: JsonDict = {}
        computed_actions: JsonDict = {}

        def reveal_due(reveal_index: int) -> None:
            due_ids = sorted(
                (
                    decision_id
                    for decision_id, pending_row in pending.items()
                    if int(pending_row["due_sequence_index"]) <= reveal_index
                ),
                key=lambda decision_id: (
                    pending[decision_id]["due_sequence_index"],
                    decision_id,
                ),
            )
            for decision_id in due_ids:
                outcome = outcomes[decision_id]
                if not isinstance(outcome, Mapping):
                    counters["exact_outcome_mismatch_count"] += 1
                    del pending[decision_id]
                    continue
                action_receipt = computed_actions[decision_id]
                feedback: JsonDict = {
                    "decision_id": decision_id,
                    "decision_sequence_index": sequence[decision_id],
                    "reveal_sequence_index": reveal_index,
                    "feedback_delay_steps": reveal_index - sequence[decision_id],
                    "action_receipt_sha256": action_receipt["receipt_sha256"],
                    "outcome_identity": outcome["outcome_identity"],
                    "exact_outcome_hash": outcome["exact_outcome_hash"],
                    "signed_direction": int(outcome["signed_direction"]),
                    "revealed_after_action_receipt": True,
                }
                feedback["feedback_receipt_sha256"] = sha256_json(feedback)
                source_feedback = feedback_sources.get(decision_id, {})
                if feedback != source_feedback:
                    counters["exact_outcome_mismatch_count"] += 1

                pending_row = pending[decision_id]
                state_before = _source_state_hash(reducer.stats, pending)
                raw_loss = _loss(
                    str(pending_row["action"]), int(outcome["signed_direction"])
                )
                write = WriteRecord(
                    write_id="pending-write",
                    decision_id=decision_id,
                    update_sequence_index=reveal_index,
                    action=str(pending_row["action"]),
                    features=tuple(float(value) for value in pending_row["features"]),
                    bounded_loss=raw_loss,
                    support_sha256=feedback["feedback_receipt_sha256"],
                )
                reducer.apply(write)
                del pending[decision_id]
                update: JsonDict = {
                    "decision_id": decision_id,
                    "action": write.action,
                    "action_receipt_sha256": pending_row[
                        "action_receipt_sha256"
                    ],
                    "feedback_receipt_sha256": feedback[
                        "feedback_receipt_sha256"
                    ],
                    "raw_loss": raw_loss,
                    "bounded_loss": raw_loss,
                    "loss_was_clamped": False,
                    "update_sequence_index": reveal_index,
                    "policy_state_sha256_before": state_before,
                    "policy_state_sha256_after": _source_state_hash(
                        reducer.stats, pending
                    ),
                }
                update["update_receipt_sha256"] = sha256_json(update)
                source_update = update_sources.get(decision_id, {})
                if update.get("update_receipt_sha256") != source_update.get(
                    "update_receipt_sha256"
                ):
                    counters["update_hash_mismatch_count"] += 1
                if (
                    update.get("policy_state_sha256_before")
                    != source_update.get("policy_state_sha256_before")
                    or update.get("policy_state_sha256_after")
                    != source_update.get("policy_state_sha256_after")
                ):
                    counters["state_hash_mismatch_count"] += 1

        for fixture_row in fixture_rows:
            decision_id = str(fixture_row["decision_id"])
            index = sequence[decision_id]
            reveal_due(index)
            features = context_features(contexts[decision_id])
            state_before = _source_state_hash(reducer.stats, pending)
            action, scores = reducer.select(features)
            receipt: JsonDict = {
                "decision_id": decision_id,
                "chosen_action": action,
                "context_sha256": sha256_json(contexts[decision_id]),
                "feature_schema_sha256": sha256_json(list(FEATURE_NAMES)),
                "policy_state_sha256_before": state_before,
                "pessimistic_scores": scores,
                "selection_reason": "minimum_pessimistic_loss",
                "unseen_context": False,
                "unseen_reasons": [],
                "due_sequence_index": index + (2 if delayed[decision_id] else 1),
                "serialized_before_feedback": True,
            }
            receipt["receipt_sha256"] = sha256_json(receipt)
            source_action = action_sources.get(decision_id, {})
            source_row = controller_rows.get(decision_id, {})
            if (
                receipt.get("receipt_sha256") != source_action.get("receipt_sha256")
                or receipt.get("context_sha256") != source_row.get("context_sha256")
                or action != source_row.get("chosen_action")
            ):
                counters["decision_hash_mismatch_count"] += 1
            if state_before != source_row.get("policy_state_sha256_before"):
                counters["state_hash_mismatch_count"] += 1
            outcome = outcomes[decision_id]
            if not isinstance(outcome, Mapping) or outcome.get(
                "exact_outcome_hash"
            ) != source_row.get("exact_outcome_hash"):
                counters["exact_outcome_mismatch_count"] += 1
            computed_actions[decision_id] = receipt
            pending[decision_id] = {
                "action": action,
                "features": list(features),
                "action_receipt_sha256": receipt["receipt_sha256"],
                "due_sequence_index": receipt["due_sequence_index"],
            }

        reveal_index = len(fixture_rows) + 1
        while pending:
            reveal_due(reveal_index)
            reveal_index += 1
        expected_count = len(fixture_rows)
        if not all(
            len(rows) == expected_count
            for rows in (
                controller_rows,
                action_sources,
                feedback_sources,
                update_sources,
            )
        ):
            counters["decision_hash_mismatch_count"] += 1
    except (KeyError, TypeError, ValueError, CounterfactualValidityError) as error:
        counters["decision_hash_mismatch_count"] += 1
        counters["error"] = type(error).__name__
    counters["passed"] = not any(
        value
        for key, value in counters.items()
        if key.endswith("_mismatch_count")
    )
    counters["imported_exp6854_aggregate_calculator"] = False
    return counters


def replay_source_hashes(
    controller: Mapping[str, Any], fixture: Mapping[str, Any]
) -> JsonDict:
    """Expose the independent receipt replay used by the precondition gate."""

    return _source_replay(controller, fixture)


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Keep expected and observed evidence for every gate result."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize all failed checks without discarding later failures."""

    failures = [dict(row) for row in checks if not row.get("passed")]
    return {
        "passed": not failures,
        "failed_check": failures[0]["check"] if failures else None,
        "failed_checks": [row["check"] for row in failures],
        "observed": failures[0]["observed"] if failures else None,
        "checks": [dict(row) for row in checks],
    }


def validate_preconditions(
    controller: Mapping[str, Any],
    fixture: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[list[JsonDict], JsonDict]:
    """Require exact source authority before any counterfactual is attempted."""

    replay = replay_source_hashes(controller, fixture)
    updates = controller.get("update_rows", [])
    write_ids = [
        str(row.get("update_receipt_sha256", ""))
        for row in updates
        if isinstance(row, Mapping)
    ]
    duplicate_count = len(write_ids) - len(set(write_ids))
    outcomes = [
        row.get("exact_later_outcome")
        for row in fixture.get("rows", [])
        if isinstance(row, Mapping)
    ]
    outcome_complete = bool(outcomes) and all(
        isinstance(row, Mapping)
        and row.get("exact_outcome_hash")
        and row.get("signed_direction") in {-1, 0, 1}
        for row in outcomes
    )
    required_contract_keys = {
        "declared_valid",
        "deletion_support",
        "substitution_support",
        "order_support",
        "coalition_support",
        "outcome_rule",
        "state_divergence_rule",
    }
    contract_valid = contract.get("declared_valid") is True and required_contract_keys <= set(
        contract
    )
    checks = [
        _check(
            "risk_sensitive_controller_complete_score",
            1,
            controller.get("risk_sensitive_controller_complete_score"),
            controller.get("risk_sensitive_controller_complete_score") == 1,
        ),
        _check(
            "declared_valid_counterfactual_contract",
            True,
            {
                "declared_valid": contract.get("declared_valid"),
                "missing_keys": sorted(required_contract_keys - set(contract)),
            },
            contract_valid,
        ),
        _check(
            "stable_decision_hashes",
            0,
            replay["decision_hash_mismatch_count"],
            replay["decision_hash_mismatch_count"] == 0,
        ),
        _check(
            "stable_state_hashes",
            0,
            replay["state_hash_mismatch_count"],
            replay["state_hash_mismatch_count"] == 0,
        ),
        _check(
            "stable_update_hashes",
            0,
            replay["update_hash_mismatch_count"],
            replay["update_hash_mismatch_count"] == 0,
        ),
        _check(
            "exact_later_outcomes",
            {"complete": True, "hash_mismatch_count": 0},
            {
                "complete": outcome_complete,
                "hash_mismatch_count": replay["exact_outcome_mismatch_count"],
            },
            outcome_complete and replay["exact_outcome_mismatch_count"] == 0,
        ),
        _check(
            "unique_write_identities",
            0,
            duplicate_count,
            bool(write_ids) and duplicate_count == 0 and all(write_ids),
        ),
    ]
    return checks, replay


def build_write_ledger(
    controller: Mapping[str, Any], fixture: Mapping[str, Any]
) -> list[WriteRecord]:
    """Join each exact update receipt to its pre-outcome feature vector."""

    contexts = {
        str(row["decision_id"]): row["decision_context"] for row in fixture["rows"]
    }
    ledger: list[WriteRecord] = []
    seen: set[str] = set()
    for row in controller["update_rows"]:
        write_id = str(row["update_receipt_sha256"])
        if not write_id or write_id in seen:
            raise CounterfactualValidityError("duplicate write identity")
        seen.add(write_id)
        decision_id = str(row["decision_id"])
        ledger.append(
            WriteRecord(
                write_id=write_id,
                decision_id=decision_id,
                update_sequence_index=int(row["update_sequence_index"]),
                action=str(row["action"]),
                features=context_features(contexts[decision_id]),
                bounded_loss=float(row["bounded_loss"]),
                support_sha256=str(row["feedback_receipt_sha256"]),
            )
        )
    return sorted(
        ledger, key=lambda item: (item.update_sequence_index, item.write_id)
    )


def validate_coalition(
    coalition: Sequence[str], eligible_write_ids: set[str]
) -> tuple[str, ...]:
    """Reject duplicate or ineligible writes before coalition replay."""

    if len(coalition) != len(set(coalition)):
        raise CounterfactualValidityError("duplicate write in coalition")
    invalid = set(coalition) - eligible_write_ids
    if invalid:
        raise CounterfactualValidityError("ineligible write in coalition")
    return tuple(coalition)


def substitution_support(target: WriteRecord, donor: WriteRecord) -> tuple[bool, str]:
    """Allow only a different observed donor with compatible transition inputs."""

    if target.write_id == donor.write_id:
        return False, "duplicate_write"
    if target.action != donor.action:
        return False, "incompatible_action"
    if target.features != donor.features:
        return False, "incompatible_features"
    if not donor.support_sha256.startswith("sha256:"):
        return False, "missing_exact_support"
    return True, "exact_observed_compatible_donor"


def coalition_credit(
    write_ids: Sequence[str],
    value_function: Callable[[frozenset[str]], float],
    *,
    random_seed: int,
    exact_limit: int = EXACT_COALITION_LIMIT,
    permutation_count: int = PERMUTATION_COUNT,
) -> tuple[list[JsonDict], JsonDict | None]:
    """Compute exact Shapley values or labeled seeded permutation intervals."""

    ordered_ids = list(write_ids)
    validate_coalition(ordered_ids, set(ordered_ids))
    count = len(ordered_ids)
    if count <= exact_limit:
        cache: dict[frozenset[str], float] = {}

        def value(coalition: frozenset[str]) -> float:
            if coalition not in cache:
                cache[coalition] = float(value_function(coalition))
            return cache[coalition]

        rows: list[JsonDict] = []
        denominator = math.factorial(count) if count else 1
        for write_id in ordered_ids:
            others = [item for item in ordered_ids if item != write_id]
            marginal = 0.0
            for size in range(len(others) + 1):
                weight = (
                    math.factorial(size)
                    * math.factorial(count - size - 1)
                    / denominator
                )
                for subset in itertools.combinations(others, size):
                    coalition = frozenset(subset)
                    marginal += weight * (
                        value(coalition | {write_id}) - value(coalition)
                    )
            rows.append(
                {
                    "write_id": write_id,
                    "marginal_value": round(marginal, 12),
                    "interval_low": round(marginal, 12),
                    "interval_high": round(marginal, 12),
                    "method": "exact_enumeration",
                    "exact": True,
                    "coalition_count": 2**count,
                }
            )
        return rows, None

    if permutation_count < 2:
        raise ValueError("permutation_count must be at least two")
    generator = random.Random(random_seed)
    samples: dict[str, list[float]] = {write_id: [] for write_id in ordered_ids}
    for _ in range(permutation_count):
        permutation = list(ordered_ids)
        generator.shuffle(permutation)
        coalition: frozenset[str] = frozenset()
        prior_value = float(value_function(coalition))
        for write_id in permutation:
            coalition = coalition | {write_id}
            next_value = float(value_function(coalition))
            samples[write_id].append(next_value - prior_value)
            prior_value = next_value
    rows = []
    maximum_width = 0.0
    for write_id in ordered_ids:
        values = samples[write_id]
        mean = statistics.fmean(values)
        standard_error = statistics.stdev(values) / math.sqrt(len(values))
        half_width = 1.96 * standard_error
        low = mean - half_width
        high = mean + half_width
        maximum_width = max(maximum_width, high - low)
        rows.append(
            {
                "write_id": write_id,
                "marginal_value": round(mean, 12),
                "interval_low": round(low, 12),
                "interval_high": round(high, 12),
                "method": "seeded_permutation_approximation",
                "exact": False,
                "permutation_count": permutation_count,
            }
        )
    receipt = {
        "method": "seeded_permutation_approximation",
        "exact": False,
        "random_seed": random_seed,
        "permutation_count": permutation_count,
        "eligible_write_count": count,
        "maximum_interval_width": round(maximum_width, 12),
        "converged": maximum_width <= 0.05,
    }
    return rows, receipt


def classify_credit(
    *,
    deletion_effect: float,
    coalition_credit: float,
    zero_headroom: bool,
    supported: bool,
) -> JsonDict:
    """Keep causal support, benefit eligibility, and semantic class separate."""

    if not supported:
        return {
            "credit_class": "unsupported",
            "causal_credit_eligible": False,
            "benefit_eligible": False,
            "benefit": None,
            "harmful_evidence": False,
            "helpful_evidence": False,
            "interaction_witness_required": False,
        }
    if zero_headroom:
        return {
            "credit_class": "zero_headroom",
            "causal_credit_eligible": True,
            "benefit_eligible": False,
            "benefit": None,
            "harmful_evidence": False,
            "helpful_evidence": False,
            "interaction_witness_required": False,
        }
    tolerance = 1e-12
    if abs(deletion_effect) <= tolerance and abs(coalition_credit) > tolerance:
        credit_class = "interaction_only"
    elif coalition_credit < -tolerance or deletion_effect < -tolerance:
        credit_class = "harmful"
    elif coalition_credit > tolerance or deletion_effect > tolerance:
        credit_class = "helpful"
    else:
        credit_class = "redundant"
    benefit = coalition_credit if credit_class != "redundant" else 0.0
    return {
        "credit_class": credit_class,
        "causal_credit_eligible": True,
        "benefit_eligible": True,
        "benefit": round(benefit, 12),
        "harmful_evidence": (
            deletion_effect < -tolerance or coalition_credit < -tolerance
        ),
        "helpful_evidence": (
            deletion_effect > tolerance or coalition_credit > tolerance
        ),
        "interaction_witness_required": credit_class == "interaction_only",
    }


def permute_placebo_contexts(
    rows: Sequence[Mapping[str, Any]], *, feature: str, seed: int
) -> list[JsonDict]:
    """Permute one pre-outcome feature while preserving row order and identities."""

    if not rows or any(feature not in row["decision_context"] for row in rows):
        raise CounterfactualValidityError("placebo feature is unavailable")
    values = [row["decision_context"][feature] for row in rows]
    random.Random(seed).shuffle(values)
    output: list[JsonDict] = []
    for row, value in zip(rows, values, strict=True):
        changed = deepcopy(dict(row))
        changed["decision_context"][feature] = value
        output.append(changed)
    return output


def _policy_metric(
    *, decision_id: str, sequence_index: int, arm: str, action: str, row: Mapping[str, Any]
) -> JsonDict:
    """Reduce one matched policy choice against the same exact source direction."""

    direction = int(row["exact_later_outcome"]["signed_direction"])
    available = action in row["available_actions"]
    loss = _loss(action, direction) if available else math.inf
    return {
        "decision_id": decision_id,
        "decision_sequence_index": sequence_index,
        "arm": arm,
        "action": action,
        "action_available": available,
        "loss": loss,
        "reward": -loss,
        "exact_outcome_hash": row["exact_later_outcome"]["exact_outcome_hash"],
    }


def _policy_controls(
    controller: Mapping[str, Any],
    fixture: Mapping[str, Any],
    ledger: Sequence[WriteRecord],
) -> tuple[list[JsonDict], JsonDict, list[JsonDict], JsonDict, JsonDict]:
    """Replay matched fixed and placebo policies without changing chronology."""

    fixture_rows = list(fixture["rows"])
    controller_rows = {
        str(row["decision_id"]): row for row in controller["rows"]
    }
    placebo_rows = permute_placebo_contexts(
        fixture_rows, feature="age", seed=RANDOM_SEED
    )
    reducer = FreshReducer()
    ledger_index = 0
    control_rows: list[JsonDict] = []
    placebo_controls: list[JsonDict] = []
    for source_row, placebo_row in zip(fixture_rows, placebo_rows, strict=True):
        decision_id = str(source_row["decision_id"])
        sequence_index = int(source_row["decision_sequence_index"])
        while (
            ledger_index < len(ledger)
            and ledger[ledger_index].update_sequence_index <= sequence_index
        ):
            reducer.apply(ledger[ledger_index])
            ledger_index += 1
        placebo_action, _ = reducer.select(
            context_features(placebo_row["decision_context"])
        )
        learned_action = str(controller_rows[decision_id]["chosen_action"])
        actions = {
            "learned_selection": learned_action,
            "no_memory": "no_memory",
            "always_memory": "verified_memory",
            "random_admission": str(
                source_row["baseline_actions"]["random_admission"]
            ),
            "abstain": "abstain",
            "placebo_context": placebo_action,
        }
        decision_metrics = {
            arm: _policy_metric(
                decision_id=decision_id,
                sequence_index=sequence_index,
                arm=arm,
                action=action,
                row=source_row,
            )
            for arm, action in actions.items()
        }
        control_rows.extend(decision_metrics.values())
        placebo_controls.append(
            {
                "decision_id": decision_id,
                "decision_sequence_index": sequence_index,
                "placebo_feature": "age",
                "observed_feature_value": source_row["decision_context"]["age"],
                "placebo_feature_value": placebo_row["decision_context"]["age"],
                "learned_action": learned_action,
                "placebo_action": placebo_action,
                "reward_delta": round(
                    decision_metrics["learned_selection"]["reward"]
                    - decision_metrics["placebo_context"]["reward"],
                    12,
                ),
                "action_availability_preserved": source_row["available_actions"]
                == placebo_row["available_actions"],
                "chronology_preserved": source_row["decision_id"]
                == placebo_row["decision_id"],
                "causal_credit_eligible": False,
            }
        )
    summary: JsonDict = {}
    for arm in (
        "learned_selection",
        "no_memory",
        "always_memory",
        "random_admission",
        "abstain",
        "placebo_context",
    ):
        rows = [row for row in control_rows if row["arm"] == arm]
        summary[arm] = {
            "decision_count": len(rows),
            "mean_loss": round(statistics.fmean(row["loss"] for row in rows), 12),
            "mean_reward": round(
                statistics.fmean(row["reward"] for row in rows), 12
            ),
            "action_counts": {
                action: sum(row["action"] == action for row in rows)
                for action in ACTIONS
            },
            "action_availability_preserved": all(
                row["action_available"] for row in rows
            ),
        }
    learned_reward = summary["learned_selection"]["mean_reward"]
    placebo_reward = summary["placebo_context"]["mean_reward"]
    no_memory_reward = summary["no_memory"]["mean_reward"]
    selection_skill = {
        "comparison": "learned_selection_minus_placebo_context",
        "mean_reward_effect": round(learned_reward - placebo_reward, 12),
        "learned_minus_no_memory": round(learned_reward - no_memory_reward, 12),
        "interpretation": (
            "Only the learned-minus-placebo term is selection skill. The broader "
            "no-memory contrast can include policy-arm and stream effects."
        ),
    }
    directions = [
        int(row["exact_later_outcome"]["signed_direction"])
        for row in fixture_rows
    ]
    stream_luck = {
        "baseline_arm": "no_memory",
        "baseline_mean_reward": no_memory_reward,
        "helpful_outcome_count": sum(value > 0 for value in directions),
        "harmful_outcome_count": sum(value < 0 for value in directions),
        "zero_headroom_outcome_count": sum(
            int(row["safe_selection_headroom"]) == 0 for row in fixture_rows
        ),
        "environmental_luck_is_not_causal_credit": True,
    }
    return control_rows, summary, placebo_controls, selection_skill, stream_luck


def _counterfactual_rows(
    fixture: Mapping[str, Any], ledger: Sequence[WriteRecord]
) -> JsonDict:
    """Run deletion, supported substitution, order, and bounded coalition audits."""

    deletion_rows: list[JsonDict] = []
    substitution_rows: list[JsonDict] = []
    order_rows: list[JsonDict] = []
    coalition_rows: list[JsonDict] = []
    approximation_receipts: list[JsonDict] = []
    unsupported_rows: list[JsonDict] = []
    fixture_rows = list(fixture["rows"])
    history: list[WriteRecord] = []

    for target in fixture_rows:
        target_index = int(target["decision_sequence_index"])
        while len(history) < len(ledger) and ledger[len(history)].update_sequence_index <= target_index:
            history.append(ledger[len(history)])
        if not history:
            unsupported_rows.append(
                {
                    "decision_id": target["decision_id"],
                    "write_id": None,
                    "counterfactual_type": "deletion",
                    "reason": "no_prior_supported_write",
                    "causal_credit_eligible": False,
                    "benefit_eligible": False,
                    "benefit": None,
                }
            )
            continue
        window = history[-COALITION_WINDOW:]
        outside = history[: -len(window)]
        base = FreshReducer()
        for write in outside:
            base.apply(write)
        by_id = {write.write_id: write for write in window}
        eligible = set(by_id)
        target_features = context_features(target["decision_context"])
        direction = int(target["exact_later_outcome"]["signed_direction"])
        zero_headroom = int(target["safe_selection_headroom"]) == 0

        def evaluate(coalition: frozenset[str]) -> tuple[float, str, str]:
            validate_coalition(list(coalition), eligible)
            reducer = base.copy()
            chosen = sorted(
                (by_id[write_id] for write_id in coalition),
                key=lambda item: (item.update_sequence_index, item.write_id),
            )
            selection = reducer.replay(
                chosen,
                target_boundary=target_index + 1,
                target_features=target_features,
            )
            return (
                -_loss(selection.selected_action, direction),
                selection.selected_action,
                selection.state_sha256,
            )

        full = frozenset(eligible)
        factual_reward, factual_action, factual_state = evaluate(full)
        deletion_by_write: dict[str, float] = {}
        for write in window:
            counterfactual = full - {write.write_id}
            reward, action, state_hash = evaluate(counterfactual)
            effect = round(factual_reward - reward, 12)
            deletion_by_write[write.write_id] = effect
            deletion_rows.append(
                {
                    "decision_id": target["decision_id"],
                    "decision_sequence_index": target_index,
                    "write_id": write.write_id,
                    "counterfactual_type": "write_deletion",
                    "method": "exact_transition_replay",
                    "exact": True,
                    "factual_action": factual_action,
                    "counterfactual_action": action,
                    "factual_reward": factual_reward,
                    "counterfactual_reward": reward,
                    "deletion_effect": effect,
                    "factual_state_sha256": factual_state,
                    "counterfactual_state_sha256": state_hash,
                    "state_reconstructed": True,
                    "causal_credit_eligible": True,
                    "benefit_eligible": not zero_headroom,
                    "benefit": effect if not zero_headroom else None,
                    "zero_headroom": zero_headroom,
                }
            )

            compatible = [
                donor
                for donor in history
                if donor.update_sequence_index <= target_index
                and substitution_support(write, donor)[0]
            ]
            if compatible:
                donor = compatible[-1]
                substitute = replace(
                    donor,
                    write_id=f"substitute:{write.write_id}:{donor.write_id}",
                    update_sequence_index=write.update_sequence_index,
                )
                reducer = base.copy()
                substituted = [
                    substitute if item.write_id == write.write_id else item
                    for item in window
                ]
                selection = reducer.replay(
                    substituted,
                    target_boundary=target_index + 1,
                    target_features=target_features,
                )
                reward = -_loss(selection.selected_action, direction)
                substitution_rows.append(
                    {
                        "decision_id": target["decision_id"],
                        "decision_sequence_index": target_index,
                        "write_id": write.write_id,
                        "donor_write_id": donor.write_id,
                        "counterfactual_type": "write_substitution",
                        "method": "exact_observed_donor_transition",
                        "exact": True,
                        "support_reason": "exact_observed_compatible_donor",
                        "factual_action": factual_action,
                        "counterfactual_action": selection.selected_action,
                        "substitution_effect": round(factual_reward - reward, 12),
                        "state_reconstructed": True,
                        "causal_credit_eligible": True,
                        "benefit_eligible": not zero_headroom,
                        "benefit": (
                            round(factual_reward - reward, 12)
                            if not zero_headroom
                            else None
                        ),
                    }
                )
            else:
                unsupported_rows.append(
                    {
                        "decision_id": target["decision_id"],
                        "write_id": write.write_id,
                        "counterfactual_type": "write_substitution",
                        "reason": "no_exact_compatible_donor",
                        "causal_credit_eligible": False,
                        "benefit_eligible": False,
                        "benefit": None,
                    }
                )

        for first, second in zip(window, window[1:], strict=False):
            if first.update_sequence_index != second.update_sequence_index:
                continue
            ordered = list(window)
            first_index = ordered.index(first)
            second_index = ordered.index(second)
            ordered[first_index], ordered[second_index] = (
                ordered[second_index],
                ordered[first_index],
            )
            reducer = base.copy()
            selection = reducer.replay(
                ordered,
                target_boundary=target_index + 1,
                target_features=target_features,
            )
            reward = -_loss(selection.selected_action, direction)
            order_rows.append(
                {
                    "decision_id": target["decision_id"],
                    "decision_sequence_index": target_index,
                    "write_id": first.write_id,
                    "paired_write_id": second.write_id,
                    "counterfactual_type": "exchangeable_write_order",
                    "method": "exact_same_reveal_boundary_replay",
                    "exact": True,
                    "shared_reveal_sequence_index": first.update_sequence_index,
                    "factual_action": factual_action,
                    "counterfactual_action": selection.selected_action,
                    "order_effect": round(factual_reward - reward, 12),
                    "state_reconstructed": True,
                    "causal_credit_eligible": True,
                    "benefit_eligible": not zero_headroom,
                    "benefit": (
                        round(factual_reward - reward, 12)
                        if not zero_headroom
                        else None
                    ),
                }
            )

        values, receipt = coalition_credit(
            [write.write_id for write in window],
            lambda coalition: evaluate(coalition)[0],
            random_seed=RANDOM_SEED + target_index,
            exact_limit=EXACT_COALITION_LIMIT,
            permutation_count=PERMUTATION_COUNT,
        )
        if receipt is not None:
            receipt.update(
                {
                    "decision_id": target["decision_id"],
                    "decision_sequence_index": target_index,
                }
            )
            approximation_receipts.append(receipt)
        for value in values:
            classification = classify_credit(
                deletion_effect=deletion_by_write[value["write_id"]],
                coalition_credit=float(value["marginal_value"]),
                zero_headroom=zero_headroom,
                supported=True,
            )
            coalition_rows.append(
                {
                    "decision_id": target["decision_id"],
                    "decision_sequence_index": target_index,
                    "write_id": value["write_id"],
                    "counterfactual_type": "bounded_coalition_credit",
                    **value,
                    **classification,
                    "deletion_effect": deletion_by_write[value["write_id"]],
                    "zero_headroom": zero_headroom,
                }
            )
    return {
        "deletion_rows": deletion_rows,
        "substitution_rows": substitution_rows,
        "order_rows": order_rows,
        "coalition_rows": coalition_rows,
        "approximation_receipts": approximation_receipts,
        "unsupported_counterfactual_rows": unsupported_rows,
    }


def _per_write_summary(
    ledger: Sequence[WriteRecord], coalition_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Reduce row evidence per write while retaining harmful and interaction signs."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in coalition_rows:
        grouped[str(row["write_id"])].append(row)
    summary: list[JsonDict] = []
    witnesses: list[JsonDict] = []
    for write in ledger:
        rows = grouped.get(write.write_id, [])
        if not rows:
            summary.append(
                {
                    "write_id": write.write_id,
                    "source_decision_id": write.decision_id,
                    "credit_class": "unsupported",
                    "supported_target_count": 0,
                    "causal_credit_eligible": False,
                    "benefit_eligible": False,
                    "benefit": None,
                }
            )
            continue
        eligible_rows = [row for row in rows if row["benefit_eligible"]]
        deletion_effect = round(
            sum(float(row["deletion_effect"]) for row in eligible_rows), 12
        )
        shapley = round(
            sum(float(row["marginal_value"]) for row in eligible_rows), 12
        )
        classification = classify_credit(
            deletion_effect=deletion_effect,
            coalition_credit=shapley,
            zero_headroom=not eligible_rows,
            supported=True,
        )
        result = {
            "write_id": write.write_id,
            "source_decision_id": write.decision_id,
            "supported_target_count": len(rows),
            "benefit_eligible_target_count": len(eligible_rows),
            "aggregate_deletion_effect": deletion_effect,
            "aggregate_coalition_credit": shapley,
            **classification,
        }
        row_classes = {str(row["credit_class"]) for row in eligible_rows}
        if "harmful" in row_classes:
            result["credit_class"] = "harmful"
        elif "interaction_only" in row_classes:
            result["credit_class"] = "interaction_only"
        elif "helpful" in row_classes:
            result["credit_class"] = "helpful"
        else:
            result["credit_class"] = "redundant"
        result["harmful_evidence"] = any(
            bool(row["harmful_evidence"]) for row in eligible_rows
        )
        result["helpful_evidence"] = any(
            bool(row["helpful_evidence"]) for row in eligible_rows
        )
        summary.append(result)
        interaction_rows = [
            row for row in eligible_rows if row["credit_class"] == "interaction_only"
        ]
        if interaction_rows:
            witnesses.append(
                {
                    "write_id": write.write_id,
                    "decision_ids": [row["decision_id"] for row in interaction_rows],
                    "deletion_effect": deletion_effect,
                    "coalition_credit": shapley,
                    "method_rows": [row["method"] for row in interaction_rows],
                }
            )
    return summary, witnesses


def compute_terminal_gates(
    *,
    planned_decisions: int,
    completed_decisions: int,
    supported_replay_count: int,
    causal_benefit_count: int,
    validation_passed: bool,
) -> JsonDict:
    """Keep audit execution completeness separate from eligible causal benefit."""

    complete = int(
        validation_passed
        and planned_decisions > 0
        and completed_decisions == planned_decisions
        and supported_replay_count > 0
    )
    return {
        "counterfactual_memory_audit_complete_score": complete,
        "causal_memory_credit_eligible_score": int(
            complete == 1 and causal_benefit_count > 0
        ),
    }


def _field_principles() -> JsonDict:
    """Explain why each artifact field exists in plain language."""

    purposes = {
        "schema": "A versioned schema prevents silent reinterpretation.",
        "experiment_id": "A stable identity binds the artifact to this task.",
        "title": "The title names the causal memory audit.",
        "status": "Status distinguishes completed and blocked execution.",
        "run_date": "The requested date fixes the evidence boundary.",
        "openspec_requirement_ids": "Requirement IDs connect rows to tested rules.",
        "field_principles": "Each top-level field has one stated purpose.",
        "preconditions_checked": "Invalid source authority blocks replay.",
        "inference_substrate": "The CPU declaration prevents an implied model call.",
        "duration_s": "Measured wall time shows that replay executed.",
        "source_artifact_hashes": "Hashes bind the exact source evidence.",
        "random_seed": "One seed controls placebo and bounded permutations.",
        "reproducibility_checksum": "The checksum binds stable replay content.",
        "rows": "Normalized rows expose one decision, write, arm, and metric.",
        "valid_counterfactual_contract": "The contract states which transitions are valid.",
        "deletion_rows": "Deletion rows test one prior write at a time.",
        "substitution_rows": "Substitution rows use exact compatible donor writes.",
        "order_rows": "Order rows swap only exchangeable reveal-boundary writes.",
        "coalition_rows": "Coalition rows report exact or bounded marginal credit.",
        "approximation_receipts": "Receipts keep approximations distinct from exact values.",
        "unsupported_counterfactual_rows": "Unsupported rows prevent invented outcomes.",
        "placebo_control_rows": "Placebo rows isolate noncausal context effects.",
        "policy_control_rows": "Matched policy rows preserve decisions and chronology.",
        "policy_control_summary": "Separate arm summaries prevent pooled control claims.",
        "per_write_credit_summary": "Each source write retains its own sign and support.",
        "harmful_write_count": "The count keeps harmful evidence visible.",
        "redundant_write_count": "The count reports supported writes with zero effect.",
        "interaction_witnesses": "Witnesses show effects visible only in coalitions.",
        "selection_skill_effect": "Learned-minus-placebo isolates selection skill.",
        "stream_luck_effect": "The no-memory baseline records stream composition.",
        "source_replay_validation": "Fresh hashes test the producer independently.",
        "counterfactual_memory_audit_complete_score": "Execution completeness excludes benefit.",
        "causal_memory_credit_eligible_score": "Eligibility requires supported nonzero-headroom credit.",
        "gate_check_summary": "Every failed check keeps its observed value.",
        "replay_commands": "Commands let another operator repeat all checks.",
        "verifier_is_oracle": "False because exact source outcomes define loss.",
        "verdict_class": "A closed class prevents causal overclaiming.",
        "honest_verdict": "A terminal row-supported verdict closes the task.",
    }
    return purposes


def _normalized_rows(
    counterfactuals: Mapping[str, Sequence[Mapping[str, Any]]],
    policy_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Normalize detailed evidence without replacing the detailed source rows."""

    rows: list[JsonDict] = []
    mappings = (
        ("deletion_rows", "deletion_effect"),
        ("substitution_rows", "substitution_effect"),
        ("order_rows", "order_effect"),
        ("coalition_rows", "marginal_value"),
    )
    for field, metric in mappings:
        for source in counterfactuals[field]:
            rows.append(
                {
                    "decision_id": source["decision_id"],
                    "write_id": source["write_id"],
                    "arm": "learned_selection",
                    "counterfactual_metric": metric,
                    "metric_value": source[metric],
                    "method": source["method"],
                }
            )
    for source in policy_rows:
        rows.append(
            {
                "decision_id": source["decision_id"],
                "write_id": None,
                "arm": source["arm"],
                "counterfactual_metric": "policy_reward",
                "metric_value": source["reward"],
                "method": "matched_chronological_policy_replay",
            }
        )
    return rows


def _base_artifact(
    repo_root: Path,
    *,
    run_date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> JsonDict:
    """Create fields shared by blocked and completed terminal artifacts."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "title": "Counterfactual memory credit audit",
        "status": "blocked",
        "run_date": run_date,
        "openspec_requirement_ids": list(OPEN_SPEC_IDS),
        "field_principles": _field_principles(),
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": source_artifact_hashes(repo_root),
        "random_seed": RANDOM_SEED,
        "rows": [],
        "valid_counterfactual_contract": deepcopy(dict(contract)),
        "deletion_rows": [],
        "substitution_rows": [],
        "order_rows": [],
        "coalition_rows": [],
        "approximation_receipts": [],
        "unsupported_counterfactual_rows": [],
        "placebo_control_rows": [],
        "policy_control_rows": [],
        "policy_control_summary": {},
        "per_write_credit_summary": [],
        "harmful_write_count": 0,
        "redundant_write_count": 0,
        "interaction_witnesses": [],
        "selection_skill_effect": {},
        "stream_luck_effect": {},
        "source_replay_validation": {},
        "counterfactual_memory_audit_complete_score": 0,
        "causal_memory_credit_eligible_score": 0,
        "gate_check_summary": _gate_summary(checks),
        "replay_commands": [
            ".venv/bin/python scripts/experiments/experiment_6855_counterfactual_memory_credit_audit.py --date 20260901",
            ".venv/bin/pytest tests/python/test_experiment_6855_counterfactual_memory_credit_audit.py -q",
            ".venv/bin/python scripts/adversarial_verify.py results/experiment_6855_counterfactual_memory_credit_audit.json",
        ],
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }


def build_artifact(
    repo_root: Path,
    *,
    run_date: str,
    duration_s: float,
    controller: Mapping[str, Any] | None = None,
    fixture: Mapping[str, Any] | None = None,
    contract: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build a blocked or complete audit from exact source artifacts."""

    controller_source = (
        dict(controller)
        if controller is not None
        else load_source(repo_root / CONTROLLER_RELATIVE_PATH)
    )
    fixture_source = (
        dict(fixture)
        if fixture is not None
        else load_source(repo_root / FIXTURE_RELATIVE_PATH)
    )
    active_contract = dict(contract or VALID_COUNTERFACTUAL_CONTRACT)
    checks, source_replay = validate_preconditions(
        controller_source, fixture_source, active_contract
    )
    artifact = _base_artifact(
        repo_root,
        run_date=run_date,
        duration_s=duration_s,
        checks=checks,
        contract=active_contract,
    )
    artifact["source_replay_validation"] = source_replay
    if not artifact["gate_check_summary"]["passed"]:
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    ledger = build_write_ledger(controller_source, fixture_source)
    counterfactuals = _counterfactual_rows(fixture_source, ledger)
    summary, interactions = _per_write_summary(
        ledger, counterfactuals["coalition_rows"]
    )
    policy_rows, policy_summary, placebo_rows, selection_skill, stream_luck = (
        _policy_controls(controller_source, fixture_source, ledger)
    )
    gates = compute_terminal_gates(
        planned_decisions=len(fixture_source["rows"]),
        completed_decisions=len(
            {row["decision_id"] for row in policy_rows if row["arm"] == "learned_selection"}
        ),
        supported_replay_count=len(counterfactuals["deletion_rows"]),
        causal_benefit_count=sum(
            row["benefit_eligible"] and abs(float(row["marginal_value"])) > 1e-12
            for row in counterfactuals["coalition_rows"]
        ),
        validation_passed=True,
    )
    artifact.update(counterfactuals)
    artifact.update(gates)
    artifact.update(
        {
            "status": "complete",
            "policy_control_rows": policy_rows,
            "policy_control_summary": policy_summary,
            "placebo_control_rows": placebo_rows,
            "per_write_credit_summary": summary,
            "harmful_write_count": sum(
                bool(row.get("harmful_evidence")) for row in summary
            ),
            "redundant_write_count": sum(
                row["credit_class"] == "redundant" for row in summary
            ),
            "interaction_witnesses": interactions,
            "selection_skill_effect": selection_skill,
            "stream_luck_effect": stream_luck,
        }
    )
    artifact["rows"] = _normalized_rows(counterfactuals, policy_rows)
    causal_score = artifact["causal_memory_credit_eligible_score"]
    harmful_count = artifact["harmful_write_count"]
    if harmful_count:
        artifact["verdict_class"] = "partial"
        artifact["honest_verdict"] = (
            "complete_partial_counterfactual_memory_credit_harmful_writes_present"
        )
    elif causal_score:
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = (
            "complete_positive_counterfactual_memory_credit_supported"
        )
    else:
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = (
            "complete_null_counterfactual_memory_credit_no_eligible_effect"
        )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable content while excluding wall time and the checksum itself."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Reject missing fields, bad declarations, and inconsistent terminal gates."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("invalid verdict_class")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict must start with complete_")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if artifact.get("counterfactual_memory_audit_complete_score") == 1:
        if not artifact.get("rows") or not artifact.get("deletion_rows"):
            errors.append("complete score contradicts replay rows")
        if not artifact.get("gate_check_summary", {}).get("passed"):
            errors.append("complete score contradicts failed preconditions")
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Validate before writing so malformed terminal evidence cannot land."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(artifact))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic audit and write only the requested output path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    artifact = build_artifact(
        REPO_ROOT,
        run_date=args.date,
        duration_s=0.0,
    )
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    write_artifact(args.output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
