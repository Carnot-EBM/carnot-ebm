"""Fail-closed FR-11 shadow adapter for exact-gated verify/repair receipts.

Spec refs: REQ-LEARN-5640,
SCENARIO-LEARN-5640-EQUIVALENCE,
SCENARIO-LEARN-5640-SHADOW,
SCENARIO-LEARN-5640-REPLAY.

The adapter records what the audited conformal KAN controller would recommend
while the exact verifier remains the only authority. It never changes the
caller-visible verification verdict and never writes model weights. Its durable
state is intentionally plain JSONL plus an atomically replaced checkpoint so a
partial write cannot publish candidate state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


ACTIONS = ("retain", "smooth", "reset", "adapt", "abstain")
GENESIS_HASH = "sha256:" + "0" * 64
CHECKPOINT_SCHEMA = "carnot.fr11_shadow_adapter.checkpoint.v1"


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible values with stable bytes for hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for JSON-compatible data."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _row_hash(row: Mapping[str, Any]) -> str:
    material = dict(row)
    material.pop("ledger_hash", None)
    return sha256_json(material)


def _checkpoint_hash(checkpoint: Mapping[str, Any]) -> str:
    material = dict(checkpoint)
    material.pop("checkpoint_hash", None)
    return sha256_json(material)


def load_ledger(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load an append-only JSONL ledger, rejecting malformed rows.

    Missing ledgers are treated as empty because a disabled or never-used
    shadow adapter must be a no-op for callers.
    """

    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"ledger row {line_number} is not an object")
        if parsed.get("ledger_hash") != _row_hash(parsed):
            raise ValueError(f"ledger row {line_number} hash mismatch")
        rows.append(parsed)
    return rows


def ledger_lineage_complete(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Return true when every row points to the previous row hash."""

    previous = GENESIS_HASH
    for index, row in enumerate(rows):
        if row.get("sequence") != index:
            return False
        if row.get("previous_ledger_hash") != previous:
            return False
        if row.get("ledger_hash") != _row_hash(row):
            return False
        previous = str(row["ledger_hash"])
    return True


def _validate_checkpoint(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("checkpoint is not an object")
    if parsed.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema")
    if parsed.get("checkpoint_hash") != _checkpoint_hash(parsed):
        raise ValueError("checkpoint hash mismatch")
    return parsed


@dataclass(frozen=True)
class ExactVerificationReceipt:
    """Exact verifier receipt consumed by the shadow adapter.

    `exact_valid` is deliberately tri-state. True means the exact verifier
    accepted the state, False means it rejected it, and None means the state is
    unsupported or not exact enough to learn from. The adapter abstains for the
    latter two states.
    """

    receipt_id: str
    input_payload: Mapping[str, Any]
    checkpoint_parent: str = GENESIS_HASH
    conformal_action_set: Sequence[str] = ("retain", "abstain")
    exact_valid: bool | None = None
    delayed_label: bool = False
    poison: bool = False
    rollback_required: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def input_hash(self) -> str:
        """Hash the logical input, not the adapter state path."""

        return sha256_json(
            {
                "receipt_id": self.receipt_id,
                "input_payload": self.input_payload,
            }
        )

    @classmethod
    def from_verification_result(
        cls,
        *,
        question: str,
        response: str,
        domain: str | None,
        result: Any,
    ) -> "ExactVerificationReceipt":
        """Build a receipt from `VerifyRepairPipeline.verify(...)` output."""

        unsupported = bool(result.certificate.get("error_type") or result.certificate.get("error"))
        exact_valid = None if unsupported else bool(result.verified)
        action_set: tuple[str, ...] = ("retain", "smooth", "abstain") if exact_valid else ("abstain",)
        return cls(
            receipt_id=sha256_json(
                {
                    "question": question,
                    "response": response,
                    "domain": domain,
                    "n_constraints": result.certificate.get("n_constraints"),
                    "n_violations": result.certificate.get("n_violations"),
                }
            ),
            input_payload={
                "question": question,
                "response": response,
                "domain": domain,
                "verified": bool(result.verified),
                "energy": float(result.energy),
            },
            checkpoint_parent=str(result.certificate.get("checkpoint_hash") or GENESIS_HASH),
            conformal_action_set=action_set,
            exact_valid=exact_valid,
            metadata={
                "mode": result.mode,
                "n_constraints": result.certificate.get("n_constraints", 0),
                "n_violations": result.certificate.get("n_violations", 0),
            },
        )


@dataclass(frozen=True)
class ShadowDecision:
    """One auditable conformal recommendation made in shadow mode."""

    row: Mapping[str, Any]

    @property
    def recommendation(self) -> str:
        return str(self.row["recommendation"])

    @property
    def rollback_reason(self) -> str:
        return str(self.row["rollback_reason"])

    @property
    def unsafe_update_accepted(self) -> bool:
        return bool(self.row["unsafe_update_accepted"])

    @property
    def checkpoint_parent(self) -> str:
        return str(self.row["checkpoint_parent"])

    @property
    def corrupted_checkpoint(self) -> bool:
        return bool(self.row["corrupted_checkpoint"])

    def to_certificate(self) -> dict[str, Any]:
        """Return the small certificate payload attached to verification results."""

        return {
            "input_hash": self.row["input_hash"],
            "checkpoint_parent": self.row["checkpoint_parent"],
            "conformal_action_set": list(self.row["conformal_action_set"]),
            "recommendation": self.row["recommendation"],
            "exact_disposition": self.row["exact_disposition"],
            "rollback_reason": self.row["rollback_reason"],
            "ledger_hash": self.row["ledger_hash"],
            "checkpoint_hash": self.row["checkpoint_hash"],
        }


class FR11ShadowAdapter:
    """Append-only exact-gated shadow adapter.

    The adapter can be constructed disabled. In that state `observe()` returns
    None and does not create files, matching the production rollout contract.
    """

    def __init__(
        self,
        *,
        ledger_path: str | os.PathLike[str],
        checkpoint_path: str | os.PathLike[str] | None = None,
        enabled: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.ledger_path = Path(ledger_path)
        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else self.ledger_path.with_suffix(".checkpoint.json")
        )
        self._rows = load_ledger(self.ledger_path) if self.ledger_path.exists() else []
        if not ledger_lineage_complete(self._rows):
            raise ValueError("ledger lineage incomplete")
        self._seen_input_hashes = {str(row["input_hash"]) for row in self._rows}
        self._checkpoint_corrupted = False
        self._checkpoint_hash = self._rows[-1]["checkpoint_hash"] if self._rows else GENESIS_HASH
        if self.checkpoint_path.exists():
            try:
                checkpoint = _validate_checkpoint(self.checkpoint_path)
                if checkpoint.get("ledger_tail_hash") != (
                    self._rows[-1]["ledger_hash"] if self._rows else GENESIS_HASH
                ):
                    raise ValueError("checkpoint ledger tail mismatch")
                self._checkpoint_hash = str(checkpoint["checkpoint_hash"])
            except (OSError, ValueError, json.JSONDecodeError):
                self._checkpoint_corrupted = True
                self._checkpoint_hash = self._rows[-1]["checkpoint_hash"] if self._rows else GENESIS_HASH

    @property
    def decision_count(self) -> int:
        """Number of durable decisions currently visible in the ledger."""

        return len(self._rows)

    def observe(self, receipt: ExactVerificationReceipt) -> ShadowDecision | None:
        """Record one exact verifier receipt and return the shadow decision."""

        if not self.enabled:
            return None

        action_set = self._safe_action_set(receipt)
        exact_disposition = self._exact_disposition(receipt)
        rollback_reason = ""
        recommendation = self._choose_recommendation(action_set)
        corrupted_checkpoint = self._checkpoint_corrupted

        if corrupted_checkpoint:
            recommendation = "abstain"
            action_set = ("abstain",)
            rollback_reason = "corrupted_checkpoint_recovered"
        elif receipt.input_hash in self._seen_input_hashes:
            recommendation = "abstain"
            action_set = ("abstain",)
            rollback_reason = "duplicate_delivery"
        elif receipt.delayed_label:
            recommendation = "abstain"
            action_set = ("abstain",)
            rollback_reason = "delayed_label_pending"
        elif receipt.poison:
            recommendation = "abstain"
            action_set = ("abstain",)
            rollback_reason = "poison_rejected"
        elif receipt.rollback_required:
            recommendation = "abstain"
            action_set = ("abstain",)
            rollback_reason = "rollback_required"
        elif receipt.exact_valid is None:
            recommendation = "abstain"
            action_set = ("abstain",)
            rollback_reason = "unsupported_exact_state"
        elif receipt.exact_valid is False:
            recommendation = "abstain"
            action_set = ("abstain",)
            rollback_reason = "exact_rejection_authoritative"
        elif recommendation == "abstain":
            rollback_reason = "conformal_action_set_abstained"

        previous_ledger_hash = self._rows[-1]["ledger_hash"] if self._rows else GENESIS_HASH
        row: dict[str, Any] = {
            "sequence": len(self._rows),
            "receipt_id": receipt.receipt_id,
            "input_hash": receipt.input_hash,
            "checkpoint_parent": self._checkpoint_hash or receipt.checkpoint_parent,
            "receipt_checkpoint_parent": receipt.checkpoint_parent,
            "conformal_action_set": list(action_set),
            "recommendation": recommendation,
            "exact_disposition": exact_disposition,
            "rollback_reason": rollback_reason,
            "duplicate_delivery": receipt.input_hash in self._seen_input_hashes,
            "delayed_label": bool(receipt.delayed_label),
            "poison": bool(receipt.poison),
            "rollback_required": bool(receipt.rollback_required),
            "corrupted_checkpoint": corrupted_checkpoint,
            "unsafe_update_accepted": bool(receipt.exact_valid is False and recommendation != "abstain"),
            "previous_ledger_hash": previous_ledger_hash,
            "checkpoint_hash": "",
            "ledger_hash": "",
            "metadata": dict(receipt.metadata),
        }
        row["checkpoint_hash"] = self._next_checkpoint_hash(row)
        row["ledger_hash"] = _row_hash(row)

        self._append_row(row)
        self._rows.append(row)
        self._seen_input_hashes.add(receipt.input_hash)
        self._commit_checkpoint(row)
        self._checkpoint_hash = str(row["checkpoint_hash"])
        self._checkpoint_corrupted = False
        return ShadowDecision(row)

    @staticmethod
    def _safe_action_set(receipt: ExactVerificationReceipt) -> tuple[str, ...]:
        actions: list[str] = []
        for action in receipt.conformal_action_set:
            if action in ACTIONS and action not in actions:
                actions.append(action)
        return tuple(actions or ["abstain"])

    @staticmethod
    def _choose_recommendation(action_set: Sequence[str]) -> str:
        for action in ("retain", "smooth", "reset", "adapt", "abstain"):
            if action in action_set:
                return action
        return "abstain"

    @staticmethod
    def _exact_disposition(receipt: ExactVerificationReceipt) -> str:
        if receipt.exact_valid is True:
            return "accept"
        if receipt.exact_valid is False:
            return "reject"
        return "unsupported"

    def _next_checkpoint_hash(self, row: Mapping[str, Any]) -> str:
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "last_sequence": row["sequence"],
            "decision_count": int(row["sequence"]) + 1,
            "ledger_tail_hash": row["ledger_hash"] or _row_hash(row),
            "checkpoint_parent": row["checkpoint_parent"],
            "seen_input_hashes": sorted((*self._seen_input_hashes, str(row["input_hash"]))),
            "checkpoint_hash": "",
        }
        return _checkpoint_hash(checkpoint)

    def _append_row(self, row: Mapping[str, Any]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(dict(row)) + "\n")

    def _commit_checkpoint(self, row: Mapping[str, Any]) -> None:
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "last_sequence": row["sequence"],
            "decision_count": int(row["sequence"]) + 1,
            "ledger_tail_hash": row["ledger_hash"],
            "checkpoint_parent": row["checkpoint_parent"],
            "seen_input_hashes": sorted(self._seen_input_hashes),
            "checkpoint_hash": "",
        }
        checkpoint["checkpoint_hash"] = _checkpoint_hash(checkpoint)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.checkpoint_path.with_name(self.checkpoint_path.name + ".tmp")
        tmp_path.write_text(canonical_json(checkpoint), encoding="utf-8")
        os.replace(tmp_path, self.checkpoint_path)
