"""Bounded online policy state for verified-memory selection.

The controller learns only a small external decision policy. It never receives
model weights, token states, or later outcomes while it chooses an action.
This keeps online adaptation reversible without changing the foundation model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


JsonDict = dict[str, Any]

ACTIONS = ("verified_memory", "no_memory", "abstain")
TIE_BREAK_ORDER = ("abstain", "no_memory", "verified_memory")
KNOWN_FAMILIES = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
KNOWN_CORRECTION_STATUSES = (
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
    *(f"family:{family}" for family in KNOWN_FAMILIES),
    *(f"correction:{status}" for status in KNOWN_CORRECTION_STATUSES),
)
DENIED_CONTEXT_FIELDS = frozenset(
    {
        "exact_later_outcome",
        "exact_outcome",
        "exact_outcome_hash",
        "outcome_identity",
        "signed_direction",
        "memory_effect_class",
        "safe_selection_headroom",
        "predicted_direction",
        "residual_pressure",
        "decision_correct",
        "regret",
    }
)
STATE_SCHEMA = "carnot.risk_sensitive_contextual_bandit_state.v1"


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable bytes so checkpoint identity does not depend on formatting."""

    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 digest with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash structured data through the stable JSON representation."""

    return sha256_bytes(canonical_json_bytes(value))


class FeedbackError(ValueError):
    """Signal feedback that cannot match one frozen pending decision."""


class CheckpointCorruptionError(ValueError):
    """Signal checkpoint bytes that fail structure or content validation."""


@dataclass(frozen=True)
class RiskMatrix:
    """Map an exact memory effect to fixed asymmetric action losses."""

    risk_ratio: float = 3.0
    missed_reuse_loss: float = 1.0
    helpful_abstention_loss: float = 0.6
    other_abstention_loss: float = 0.1
    neutral_memory_cost: float = 0.25

    def __post_init__(self) -> None:
        """Reject a ratio that would reverse the declared safety asymmetry."""

        if not math.isfinite(self.risk_ratio) or self.risk_ratio <= self.missed_reuse_loss:
            raise ValueError("risk_ratio must exceed missed_reuse_loss")
        if self.risk_ratio <= self.helpful_abstention_loss:
            raise ValueError("risk_ratio must exceed helpful_abstention_loss")

    def loss(self, action: str, direction: int) -> float:
        """Return bounded loss from the exact observed memory-effect direction."""

        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action}")
        if direction not in {-1, 0, 1}:
            raise ValueError("direction must be -1, 0, or 1")
        if direction > 0:
            return {
                "verified_memory": 0.0,
                "no_memory": self.missed_reuse_loss,
                "abstain": self.helpful_abstention_loss,
            }[action]
        if direction < 0:
            return {
                "verified_memory": self.risk_ratio,
                "no_memory": 0.0,
                "abstain": self.other_abstention_loss,
            }[action]
        return {
            "verified_memory": self.neutral_memory_cost,
            "no_memory": 0.0,
            "abstain": self.other_abstention_loss,
        }[action]

    def as_dict(self) -> JsonDict:
        """Expose the full matrix so an artifact can audit every action cost."""

        return {
            "risk_ratio": self.risk_ratio,
            "principle": (
                "Harmful memory injection costs more than missed reuse or helpful-case "
                "abstention."
            ),
            "losses": {
                label: {action: self.loss(action, direction) for action in ACTIONS}
                for label, direction in (("helpful", 1), ("harmful", -1), ("ambiguous", 0))
            },
        }


@dataclass(frozen=True)
class PolicyConfig:
    """Fix all state and confidence bounds used by the online policy."""

    risk_ratio: float = 3.0
    ridge: float = 1.0
    confidence_scale: float = 0.12
    max_updates_per_action: int = 256
    max_pending: int = 4
    max_state_bytes: int = 16_384

    def __post_init__(self) -> None:
        """Reject invalid limits before they can enter a checkpoint."""

        if self.risk_ratio <= 1.0 or self.ridge <= 0.0 or self.confidence_scale < 0.0:
            raise ValueError("policy numeric bounds must be positive")
        if self.max_updates_per_action < 1 or self.max_pending < 1:
            raise ValueError("policy capacities must be positive")
        if self.max_state_bytes < 1024:
            raise ValueError("max_state_bytes is too small")


@dataclass(frozen=True)
class ContextEncoding:
    """Hold the fixed vector plus reasons for conservative abstention."""

    features: tuple[float, ...]
    unseen: bool
    unseen_reasons: tuple[str, ...]


def _walk_keys(value: Any) -> set[str]:
    """Collect nested keys so outcome fields cannot hide inside containers."""

    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            found.add(str(key))
            found.update(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_walk_keys(child))
    return found


def encode_context(context: Mapping[str, Any]) -> ContextEncoding:
    """Encode only the fixed pre-outcome schema and flag unknown inputs."""

    reasons: set[str] = set()

    def bounded_number(name: str, default: float, *, upper: float = 1.0) -> float:
        value = context.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            reasons.add(name)
            return default
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > upper:
            reasons.add(name)
            return default
        return number

    if _walk_keys(context) & DENIED_CONTEXT_FIELDS:
        reasons.add("denied_field")
    relevance = bounded_number("relevance", 0.0)
    uncertainty = bounded_number("uncertainty", 1.0)
    false_positive_risk = bounded_number("false_positive_risk", 1.0)
    compatibility = context.get("exact_compatibility") is True
    if not compatibility:
        reasons.add("exact_compatibility")

    age_value = context.get("age", 0)
    if isinstance(age_value, bool) or not isinstance(age_value, (int, float)) or age_value < 0:
        reasons.add("age")
        age_value = 0
    age = min(float(age_value), 8.0) / 8.0

    capacity = context.get("capacity", {})
    if not isinstance(capacity, Mapping):
        reasons.add("capacity")
        capacity = {}
    budget_value = capacity.get("budget", 0)
    if isinstance(budget_value, bool) or not isinstance(budget_value, (int, float)) or budget_value < 0:
        reasons.add("capacity")
        budget_value = 0
    budget = min(float(budget_value), 4.0) / 4.0
    admitted = capacity.get("factual_read_admitted") is True
    if not admitted:
        reasons.add("capacity")

    family = str(context.get("family", ""))
    if family not in KNOWN_FAMILIES:
        reasons.add("family")
    status = str(context.get("correction_status", ""))
    if status not in KNOWN_CORRECTION_STATUSES:
        reasons.add("correction_status")

    features = (
        1.0,
        relevance,
        uncertainty,
        float(compatibility),
        age,
        budget,
        float(admitted),
        false_positive_risk,
        *(float(family == known) for known in KNOWN_FAMILIES),
        *(float(status == known) for known in KNOWN_CORRECTION_STATUSES),
    )
    return ContextEncoding(features, bool(reasons), tuple(sorted(reasons)))


def deterministic_argmin(scores: Mapping[str, float]) -> str:
    """Choose the smallest score and use the declared safety order for ties."""

    if not scores:
        raise ValueError("no action scores")
    available = [action for action in TIE_BREAK_ORDER if action in scores]
    if not available:
        raise ValueError("no known action scores")
    return min(available, key=lambda action: (float(scores[action]), TIE_BREAK_ORDER.index(action)))


class RiskSensitiveContextualBandit:
    """Learn diagonal fixed-feature loss estimates with bounded sufficient state."""

    _PRIOR_LOSS = {"verified_memory": 0.72, "no_memory": 0.30, "abstain": 0.25}

    def __init__(self, config: PolicyConfig | None = None) -> None:
        """Create a clean controller with fixed-size statistics and no history."""

        self.config = config or PolicyConfig()
        self._stats = {
            action: {
                "precision": [self.config.ridge] * len(FEATURE_NAMES),
                "loss_sum": [
                    self.config.ridge * self._PRIOR_LOSS[action]
                ]
                * len(FEATURE_NAMES),
                "updates": 0,
            }
            for action in ACTIONS
        }
        self._pending: dict[str, JsonDict] = {}

    @property
    def pending_count(self) -> int:
        """Return the number of frozen decisions awaiting exact feedback."""

        return len(self._pending)

    @property
    def action_stats(self) -> Mapping[str, JsonDict]:
        """Expose bounded sufficient statistics for tests and artifact receipts."""

        return self._stats

    def is_clean(self) -> bool:
        """Confirm that no decision or update has changed initial policy state."""

        return not self._pending and all(row["updates"] == 0 for row in self._stats.values())

    def _state_payload(self) -> JsonDict:
        """Build the stable checkpoint payload without any unbounded history."""

        return {
            "schema": STATE_SCHEMA,
            "config": asdict(self.config),
            "feature_names": list(FEATURE_NAMES),
            "action_stats": self._stats,
            "pending": {key: self._pending[key] for key in sorted(self._pending)},
        }

    def state_hash(self) -> str:
        """Hash the active policy payload for pre-outcome action receipts."""

        return sha256_json(self._state_payload())

    def to_bytes(self) -> bytes:
        """Serialize the controller with a checksum that detects corruption."""

        payload = self._state_payload()
        envelope = {"payload": payload, "payload_sha256": sha256_json(payload)}
        checkpoint = canonical_json_bytes(envelope)
        if len(checkpoint) > self.config.max_state_bytes:
            raise CheckpointCorruptionError("controller state exceeds max_state_bytes")
        return checkpoint

    @classmethod
    def from_bytes(cls, checkpoint: bytes) -> RiskSensitiveContextualBandit:
        """Load a complete validated state or reject all checkpoint bytes."""

        try:
            envelope = json.loads(checkpoint.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CheckpointCorruptionError("checkpoint is not valid JSON") from error
        if not isinstance(envelope, dict) or not isinstance(envelope.get("payload"), dict):
            raise CheckpointCorruptionError("checkpoint envelope is invalid")
        payload = envelope["payload"]
        if envelope.get("payload_sha256") != sha256_json(payload):
            raise CheckpointCorruptionError("checkpoint payload hash mismatch")
        if payload.get("schema") != STATE_SCHEMA or payload.get("feature_names") != list(
            FEATURE_NAMES
        ):
            raise CheckpointCorruptionError("checkpoint schema mismatch")
        try:
            config = PolicyConfig(**payload["config"])
        except (KeyError, TypeError, ValueError) as error:
            raise CheckpointCorruptionError("checkpoint config is invalid") from error
        controller = cls(config)
        stats = payload.get("action_stats")
        pending = payload.get("pending")
        if not isinstance(stats, dict) or set(stats) != set(ACTIONS) or not isinstance(pending, dict):
            raise CheckpointCorruptionError("checkpoint state sections are invalid")
        if len(pending) > config.max_pending:
            raise CheckpointCorruptionError("checkpoint pending capacity exceeded")
        for action, row in stats.items():
            if not isinstance(row, dict):
                raise CheckpointCorruptionError("checkpoint action state is invalid")
            precision = row.get("precision")
            loss_sum = row.get("loss_sum")
            updates = row.get("updates")
            if (
                not isinstance(precision, list)
                or not isinstance(loss_sum, list)
                or len(precision) != len(FEATURE_NAMES)
                or len(loss_sum) != len(FEATURE_NAMES)
                or not isinstance(updates, int)
                or isinstance(updates, bool)
                or not 0 <= updates <= config.max_updates_per_action
                or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in precision)
                or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in loss_sum)
            ):
                raise CheckpointCorruptionError("checkpoint statistics are invalid")
            if any(value < config.ridge for value in precision):
                raise CheckpointCorruptionError("checkpoint precision is out of bounds")
        if any(not isinstance(key, str) or not isinstance(value, dict) for key, value in pending.items()):
            raise CheckpointCorruptionError("checkpoint pending rows are invalid")
        controller._stats = stats
        controller._pending = pending
        if len(checkpoint) > config.max_state_bytes:
            raise CheckpointCorruptionError("checkpoint exceeds max_state_bytes")
        return controller

    def save_checkpoint(self, path: Path) -> JsonDict:
        """Persist one bounded checkpoint and verify its byte round-trip."""

        checkpoint = self.to_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(checkpoint)
        loaded = path.read_bytes()
        return {
            "checkpoint_sha256": sha256_bytes(checkpoint),
            "policy_state_sha256": self.state_hash(),
            "state_bytes": len(checkpoint),
            "pending_count": self.pending_count,
            "round_trip_bytes_identical": loaded == checkpoint,
        }

    @classmethod
    def load_checkpoint(cls, path: Path) -> RiskSensitiveContextualBandit:
        """Read and validate one checkpoint file without partial recovery."""

        try:
            checkpoint = path.read_bytes()
        except OSError as error:
            raise CheckpointCorruptionError("checkpoint file is unreadable") from error
        return cls.from_bytes(checkpoint)

    def pessimistic_scores(self, encoding: ContextEncoding) -> dict[str, float]:
        """Return upper confidence loss estimates for all fixed actions."""

        weights = [abs(value) for value in encoding.features]
        denominator = sum(weights) or 1.0
        scores: dict[str, float] = {}
        for action in ACTIONS:
            row = self._stats[action]
            estimates = [
                float(total) / float(precision)
                for total, precision in zip(row["loss_sum"], row["precision"], strict=True)
            ]
            mean = sum(weight * estimate for weight, estimate in zip(weights, estimates, strict=True)) / denominator
            uncertainty = sum(
                weight / math.sqrt(float(precision))
                for weight, precision in zip(weights, row["precision"], strict=True)
            ) / denominator
            action_scale = {
                "verified_memory": self.config.risk_ratio,
                "no_memory": 1.0,
                "abstain": 0.6,
            }[action]
            scores[action] = round(mean + self.config.confidence_scale * uncertainty * action_scale, 12)
        return scores

    def freeze_action(
        self,
        decision_id: str,
        context: Mapping[str, Any],
        due_sequence_index: int,
    ) -> JsonDict:
        """Freeze and serialize an action before any exact outcome is accepted."""

        if not decision_id or decision_id in self._pending:
            raise FeedbackError("decision identity is blank or already pending")
        if len(self._pending) >= self.config.max_pending:
            raise FeedbackError("pending capacity exceeded")
        encoding = encode_context(context)
        scores = self.pessimistic_scores(encoding)
        action = "abstain" if encoding.unseen else deterministic_argmin(scores)
        reason = "conservative_unseen_context" if encoding.unseen else "minimum_pessimistic_loss"
        receipt: JsonDict = {
            "decision_id": decision_id,
            "chosen_action": action,
            "context_sha256": sha256_json(context),
            "feature_schema_sha256": sha256_json(list(FEATURE_NAMES)),
            "policy_state_sha256_before": self.state_hash(),
            "pessimistic_scores": scores,
            "selection_reason": reason,
            "unseen_context": encoding.unseen,
            "unseen_reasons": list(encoding.unseen_reasons),
            "due_sequence_index": int(due_sequence_index),
            "serialized_before_feedback": True,
        }
        receipt["receipt_sha256"] = sha256_json(receipt)
        self._pending[decision_id] = {
            "action": action,
            "features": list(encoding.features),
            "action_receipt_sha256": receipt["receipt_sha256"],
            "due_sequence_index": int(due_sequence_index),
        }
        return receipt

    def pending_due(self, sequence_index: int) -> list[str]:
        """Return stable pending identities whose reveal boundary has arrived."""

        return sorted(
            (
                decision_id
                for decision_id, row in self._pending.items()
                if int(row["due_sequence_index"]) <= sequence_index
            ),
            key=lambda decision_id: (self._pending[decision_id]["due_sequence_index"], decision_id),
        )

    def apply_bounded_loss(
        self,
        decision_id: str,
        raw_loss: float,
        *,
        feedback_receipt_sha256: str,
        update_sequence_index: int,
    ) -> JsonDict:
        """Apply one delayed loss with fixed count, value, and state bounds."""

        pending = self._pending.get(decision_id)
        if pending is None:
            raise FeedbackError("no pending action for feedback")
        if not isinstance(raw_loss, (int, float)) or isinstance(raw_loss, bool) or not math.isfinite(raw_loss):
            raise FeedbackError("feedback loss is not finite")
        state_before = self.state_hash()
        bounded_loss = min(self.config.risk_ratio, max(0.0, float(raw_loss)))
        action = str(pending["action"])
        row = self._stats[action]
        if row["updates"] >= self.config.max_updates_per_action:
            decay = (self.config.max_updates_per_action - 1) / self.config.max_updates_per_action
            prior = self._PRIOR_LOSS[action]
            row["precision"] = [
                self.config.ridge + (float(value) - self.config.ridge) * decay
                for value in row["precision"]
            ]
            row["loss_sum"] = [
                self.config.ridge * prior
                + (float(value) - self.config.ridge * prior) * decay
                for value in row["loss_sum"]
            ]
        features = [float(value) for value in pending["features"]]
        row["precision"] = [
            float(value) + feature * feature
            for value, feature in zip(row["precision"], features, strict=True)
        ]
        row["loss_sum"] = [
            float(value) + feature * feature * bounded_loss
            for value, feature in zip(row["loss_sum"], features, strict=True)
        ]
        row["updates"] = min(self.config.max_updates_per_action, int(row["updates"]) + 1)
        del self._pending[decision_id]
        update: JsonDict = {
            "decision_id": decision_id,
            "action": action,
            "action_receipt_sha256": pending["action_receipt_sha256"],
            "feedback_receipt_sha256": feedback_receipt_sha256,
            "raw_loss": float(raw_loss),
            "bounded_loss": bounded_loss,
            "loss_was_clamped": bounded_loss != float(raw_loss),
            "update_sequence_index": int(update_sequence_index),
            "policy_state_sha256_before": state_before,
            "policy_state_sha256_after": self.state_hash(),
        }
        update["update_receipt_sha256"] = sha256_json(update)
        return update
