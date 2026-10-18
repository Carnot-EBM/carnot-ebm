"""Default-off FR-11 factor-cache shadow adapter.

Spec refs: REQ-PIPELINE-6479, SCENARIO-PIPELINE-6479-SHADOW,
REQ-LEARN-6479, SCENARIO-LEARN-6479-EXACT-ADMIT,
SCENARIO-LEARN-6479-RESTART.

The adapter records what a bounded factor cache would write after exact
verification. It is a shadow channel. It never releases a pipeline answer, and
it admits cache state only after the exact checker receipt passes identity and
chronology checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from carnot.task_runtime_receipts import canonical_json, sha256_json, write_json_atomic


ADAPTER_API_VERSION = "carnot.fr11.factor_cache_shadow_adapter.v1"
CHECKPOINT_SCHEMA = "carnot.fr11.factor_cache_shadow_adapter.checkpoint.v1"
LEDGER_SCHEMA = "carnot.fr11.factor_cache_shadow_adapter.ledger_row.v1"
GENESIS_HASH = "sha256:" + "0" * 64
LEARNING_RATE = 0.25
MAX_UPDATE_MAGNITUDE = 0.25
WEIGHT_CAP = 2.0
API_METHODS = (
    "observe",
    "exact_admit",
    "propose_rank",
    "tombstone",
    "rollback",
    "save",
    "load",
    "close",
)


def adapter_api_schema_hash() -> str:
    """Hash the public lifecycle API and state schema version."""

    return sha256_json(
        {
            "api_version": ADAPTER_API_VERSION,
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "ledger_schema": LEDGER_SCHEMA,
            "methods": list(API_METHODS),
            "write_authority": "exact_checker_only",
        }
    )


def _sha_prefixed(value: Any) -> bool:
    text = str(value)
    return len(text) == 71 and text.startswith("sha256:")


def _row_hash(row: Mapping[str, Any]) -> str:
    material = dict(row)
    material.pop("row_hash", None)
    return sha256_json(material)


def _checkpoint_hash(checkpoint: Mapping[str, Any]) -> str:
    material = dict(checkpoint)
    material.pop("checkpoint_hash", None)
    return sha256_json(material)


def load_ledger(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a JSONL ledger and reject malformed hash chains."""

    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    previous = GENESIS_HASH
    for line_number, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"ledger row {line_number} is not an object")
        if parsed.get("schema") != LEDGER_SCHEMA:
            raise ValueError(f"ledger row {line_number} schema mismatch")
        if parsed.get("sequence") != len(rows):
            raise ValueError(f"ledger row {line_number} sequence mismatch")
        if parsed.get("previous_row_hash") != previous:
            raise ValueError(f"ledger row {line_number} lineage mismatch")
        if parsed.get("row_hash") != _row_hash(parsed):
            raise ValueError(f"ledger row {line_number} hash mismatch")
        rows.append(parsed)
        previous = str(parsed["row_hash"])
    return rows


@dataclass(frozen=True)
class FactorCacheEventReceipt:
    """Exact verifier receipt for one candidate factor-cache event."""

    event_id: str
    raw_hash: str
    unit_binding: str
    raw_unit_binding: str
    checker_hash: str
    exact_outcome: str
    checker_receipt: Mapping[str, Any]
    chronology_index: int
    factor_id: str
    model_confidence: float = 0.5
    selected_features: Sequence[str] = ("verified_binding",)
    cache_parent_hash: str = GENESIS_HASH
    self_signed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def exact_receipt_hash(self) -> str:
        """Hash the checker receipt that authorizes or rejects the write."""

        return sha256_json(
            {
                "checker_hash": self.checker_hash,
                "checker_receipt": dict(self.checker_receipt),
                "exact_outcome": self.exact_outcome,
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
        chronology_index: int,
        cache_parent_hash: str,
    ) -> "FactorCacheEventReceipt":
        """Convert a pipeline result into an exact-admission receipt."""

        violation_types = [item.constraint_type for item in result.violations]
        unsupported = bool(result.certificate.get("error_type") or result.certificate.get("error"))
        exact_outcome = "unsupported" if unsupported else ("pass" if result.verified else "fail")
        domain_name = domain or "auto"
        raw_payload = {
            "question": question,
            "response": response,
            "domain": domain_name,
            "verified": bool(result.verified),
            "energy": float(result.energy),
            "violations": violation_types,
        }
        unit_binding = sha256_json({"question": question, "domain": domain_name})
        factor_leaf = violation_types[0] if violation_types else "verified_binding"
        raw_hash = sha256_json(raw_payload)
        return cls(
            event_id=sha256_json({"raw_hash": raw_hash, "unit_binding": unit_binding}),
            raw_hash=raw_hash,
            unit_binding=unit_binding,
            raw_unit_binding=unit_binding,
            checker_hash=sha256_json(
                {
                    "checker": "VerifyRepairPipeline._evaluate_constraints",
                    "domain": domain_name,
                }
            ),
            exact_outcome=exact_outcome,
            checker_receipt={
                "exact_outcome": exact_outcome,
                "checker_ran_before_write": True,
                "checker_authority_passed": not unsupported,
                "result_verified": bool(result.verified),
                "n_violations": len(violation_types),
            },
            chronology_index=chronology_index,
            factor_id=f"{domain_name}:{factor_leaf}",
            model_confidence=0.8,
            selected_features=(factor_leaf,),
            cache_parent_hash=cache_parent_hash,
            self_signed=False,
            metadata={
                "mode": result.mode,
                "skipped": bool(result.skipped),
                "release_decision": bool(result.verified),
            },
        )


@dataclass(frozen=True)
class FactorCacheShadowDecision:
    """One durable shadow decision."""

    row: Mapping[str, Any]

    @property
    def exact_admission(self) -> Mapping[str, Any]:
        return self.row["exact_admission"]

    @property
    def shadow_rank(self) -> Mapping[str, Any]:
        return self.row["shadow_rank"]

    @property
    def cache_write(self) -> Mapping[str, Any]:
        return self.row["cache_write"]

    def to_certificate(self) -> dict[str, Any]:
        """Return the certificate payload attached to pipeline results."""

        return {
            "adapter_api_version": self.row["adapter_api_version"],
            "adapter_api_schema_hash": self.row["adapter_api_schema_hash"],
            "mode": "shadow",
            "release_authority": "exact_verifier",
            "event_id": self.row["event_id"],
            "raw_hash": self.row["raw_hash"],
            "unit_binding": self.row["unit_binding"],
            "chronology_index": self.row["chronology_index"],
            "shadow_rank": dict(self.row["shadow_rank"]),
            "exact_admission": dict(self.row["exact_admission"]),
            "cache_write": dict(self.row["cache_write"]),
            "row_hash": self.row["row_hash"],
        }


class FR11FactorCacheShadowAdapter:
    """Versioned exact-gated shadow adapter for factor-cache writes."""

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
        self._rows = load_ledger(self.ledger_path)
        self._sequence = len(self._rows)
        self._ledger_tail_hash = self._rows[-1]["row_hash"] if self._rows else GENESIS_HASH
        self._cache: dict[str, dict[str, Any]] = {}
        self._tombstones: dict[str, dict[str, Any]] = {}
        self._quarantine: list[dict[str, Any]] = []
        self._rollbacks: list[dict[str, Any]] = []
        self._seen_event_ids: set[str] = set()
        self._seen_raw_hashes: set[str] = set()
        self._event_cache_hashes: dict[str, str] = {}
        self._active_cache_hashes: set[str] = set()
        self._state_hash = GENESIS_HASH
        self._last_chronology_index = -1
        if self.checkpoint_path.exists():
            self._restore_checkpoint(self.checkpoint_path)
        elif self._rows:
            self._restore_from_rows(self._rows)

    @property
    def next_chronology_index(self) -> int:
        """Return the next monotonic event index expected by the adapter."""

        return self._last_chronology_index + 1

    @property
    def state_hash(self) -> str:
        """Return the current active cache state hash."""

        return self._state_hash

    @classmethod
    def load(
        cls,
        *,
        ledger_path: str | os.PathLike[str],
        checkpoint_path: str | os.PathLike[str] | None = None,
        enabled: bool = True,
    ) -> "FR11FactorCacheShadowAdapter":
        """Load adapter state from ledger and checkpoint paths."""

        return cls(ledger_path=ledger_path, checkpoint_path=checkpoint_path, enabled=enabled)

    def observe(self, receipt: FactorCacheEventReceipt) -> FactorCacheShadowDecision | None:
        """Record one shadow observation after exact verification."""

        if not self.enabled:
            return None
        shadow_rank = self.propose_rank(receipt)
        exact_admission = self.exact_admit(receipt)
        cache_write = self._cache_write_for(receipt, exact_admission)
        if exact_admission["admitted"] is True:
            self._commit_cache_write(receipt, cache_write)
        else:
            self._quarantine_receipt(receipt, exact_admission)
        self._seen_event_ids.add(receipt.event_id)
        self._seen_raw_hashes.add(receipt.raw_hash)
        self._last_chronology_index = max(self._last_chronology_index, int(receipt.chronology_index))
        row = self._base_row(
            action="observe",
            event_id=receipt.event_id,
            raw_hash=receipt.raw_hash,
            unit_binding=receipt.unit_binding,
            chronology_index=int(receipt.chronology_index),
            shadow_rank=shadow_rank,
            exact_admission=exact_admission,
            cache_write=cache_write,
            exact_receipt_hash=receipt.exact_receipt_hash,
            metadata=dict(receipt.metadata),
        )
        self._append_row(row)
        self.save()
        return FactorCacheShadowDecision(row)

    def exact_admit(self, receipt: FactorCacheEventReceipt) -> dict[str, Any]:
        """Return the exact-admission decision without mutating state."""

        reject_reason = self._reject_reason(receipt)
        admitted = reject_reason == ""
        return {
            "admitted": admitted,
            "reject_reason": reject_reason,
            "prior_exact_receipt": bool(receipt.checker_receipt),
            "exact_receipt_hash": receipt.exact_receipt_hash,
            "checker_hash": receipt.checker_hash,
            "checker_ran_before_write": receipt.checker_receipt.get("checker_ran_before_write")
            is True,
            "checker_authority_passed": receipt.checker_receipt.get("checker_authority_passed")
            is True,
            "exact_outcome": receipt.exact_outcome,
        }

    def propose_rank(self, receipt: FactorCacheEventReceipt) -> dict[str, Any]:
        """Return shadow rank advice without changing the pipeline decision."""

        if receipt.exact_outcome not in {"pass", "fail"} or not receipt.selected_features:
            return {
                "recommendation": "abstain",
                "reason": "unsupported_or_no_features",
                "proposed_rank_delta": 0.0,
            }
        sign = 1 if receipt.exact_outcome == "pass" else -1
        magnitude = self._bounded_magnitude(receipt.model_confidence)
        return {
            "recommendation": "rank",
            "reason": "shadow_only_exact_outcome_rank",
            "factor_id": receipt.factor_id,
            "features": list(receipt.selected_features),
            "proposed_rank_delta": round(sign * magnitude, 9),
        }

    def tombstone(self, event_id: str, *, reason: str) -> dict[str, Any]:
        """Tombstone an event and remove its active cache write."""

        event_text = str(event_id)
        cache_hash = self._event_cache_hashes.pop(event_text, "")
        self._active_cache_hashes.discard(cache_hash)
        for factor_id, record in list(self._cache.items()):
            if record.get("event_id") == event_text:
                self._cache.pop(factor_id)
        tombstone = {
            "event_id": event_text,
            "reason": reason,
            "rejected_cache_hash": cache_hash,
            "state_hash_before": self._state_hash,
        }
        tombstone["tombstone_hash"] = sha256_json(tombstone)
        self._tombstones[event_text] = tombstone
        self._state_hash = sha256_json(
            {
                "parent": self._state_hash,
                "tombstones": sorted(self._tombstones),
                "active_cache_hashes": sorted(self._active_cache_hashes),
            }
        )
        row = self._base_row(action="tombstone", event_id=event_text, tombstone=tombstone)
        self._append_row(row)
        self.save()
        return tombstone

    def rollback(self, *, target_cache_hash: str, reason: str) -> dict[str, Any]:
        """Rollback active cache state to a prior hash."""

        target = str(target_cache_hash)
        if target == GENESIS_HASH:
            self._cache.clear()
            self._active_cache_hashes.clear()
        else:
            self._active_cache_hashes = {target} if target in self._active_cache_hashes else set()
        rollback = {
            "target_cache_hash": target,
            "reason": reason,
            "state_hash_before": self._state_hash,
        }
        self._state_hash = target
        rollback["state_hash_after"] = self._state_hash
        rollback["rollback_hash"] = sha256_json(rollback)
        self._rollbacks.append(rollback)
        row = self._base_row(action="rollback", event_id="", rollback=rollback)
        self._append_row(row)
        self.save()
        return rollback

    def save(self) -> None:
        """Atomically persist checkpoint state."""

        if not self.enabled:
            return
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "adapter_api_version": ADAPTER_API_VERSION,
            "adapter_api_schema_hash": adapter_api_schema_hash(),
            "sequence": self._sequence,
            "ledger_tail_hash": self._ledger_tail_hash,
            "state_hash": self._state_hash,
            "cache": self._cache,
            "tombstones": self._tombstones,
            "quarantine": self._quarantine,
            "rollbacks": self._rollbacks,
            "seen_event_ids": sorted(self._seen_event_ids),
            "seen_raw_hashes": sorted(self._seen_raw_hashes),
            "event_cache_hashes": self._event_cache_hashes,
            "active_cache_hashes": sorted(self._active_cache_hashes),
            "last_chronology_index": self._last_chronology_index,
            "checkpoint_hash": "",
        }
        checkpoint["checkpoint_hash"] = _checkpoint_hash(checkpoint)
        write_json_atomic(self.checkpoint_path, checkpoint)

    def close(self) -> None:
        """Flush shadow state without changing release behavior."""

        self.save()

    def state_summary(self) -> dict[str, Any]:
        """Return a compact state receipt for tests and artifacts."""

        return {
            "state_hash": self._state_hash,
            "ledger_tail_hash": self._ledger_tail_hash,
            "sequence": self._sequence,
            "admitted_write_count": sum(
                1
                for row in self._rows
                if row.get("action") == "observe"
                and row.get("cache_write", {}).get("write_admitted") is True
            ),
            "quarantine_count": len(self._quarantine),
            "tombstone_count": len(self._tombstones),
            "rollback_count": len(self._rollbacks),
            "tombstone_hashes": sorted(row["tombstone_hash"] for row in self._tombstones.values()),
            "rollback_hashes": sorted(row["rollback_hash"] for row in self._rollbacks),
            "tombstoned_event_ids": sorted(self._tombstones),
            "active_cache_hashes": sorted(self._active_cache_hashes),
        }

    def _reject_reason(self, receipt: FactorCacheEventReceipt) -> str:
        required_present = all(
            (
                receipt.event_id,
                receipt.raw_hash,
                receipt.unit_binding,
                receipt.raw_unit_binding,
                receipt.checker_hash,
                receipt.exact_outcome,
                receipt.chronology_index is not None,
            )
        )
        if not required_present:
            return "missing_required_identity"
        if not _sha_prefixed(receipt.raw_hash) or not _sha_prefixed(receipt.checker_hash):
            return "bad_hash_format"
        if receipt.event_id in self._tombstones:
            return "tombstoned_event"
        if receipt.self_signed:
            return "self_signed_receipt"
        if receipt.event_id in self._seen_event_ids:
            return "duplicate_event_id"
        if receipt.raw_hash in self._seen_raw_hashes:
            return "duplicate_raw_hash"
        if receipt.unit_binding != receipt.raw_unit_binding:
            return "wrong_unit_binding"
        if receipt.checker_receipt.get("exact_outcome") != receipt.exact_outcome:
            return "forged_exact_outcome"
        if receipt.checker_receipt.get("checker_ran_before_write") is not True:
            return "write_before_check"
        if receipt.checker_receipt.get("checker_authority_passed") is not True:
            return "checker_authority_failed"
        if receipt.exact_outcome not in {"pass", "fail"}:
            return "unsupported_exact_outcome"
        if int(receipt.chronology_index) <= self._last_chronology_index:
            return "non_monotonic_chronology"
        if receipt.cache_parent_hash != self._state_hash:
            return "stale_cache_parent"
        return ""

    def _cache_write_for(
        self,
        receipt: FactorCacheEventReceipt,
        exact_admission: Mapping[str, Any],
    ) -> dict[str, Any]:
        sign = 1 if receipt.exact_outcome == "pass" else -1
        magnitude = self._bounded_magnitude(receipt.model_confidence)
        pre_cache_hash = self._state_hash
        post_weights = {
            factor: dict(record)
            for factor, record in self._cache.items()
            if record.get("event_id") not in self._tombstones
        }
        for feature in receipt.selected_features:
            current = float(post_weights.get(str(feature), {}).get("weight", 0.0))
            weight = max(-WEIGHT_CAP, min(WEIGHT_CAP, current + sign * magnitude))
            post_weights[str(feature)] = {
                "factor_id": receipt.factor_id,
                "feature": str(feature),
                "weight": round(weight, 9),
                "event_id": receipt.event_id,
                "exact_receipt_hash": receipt.exact_receipt_hash,
            }
        post_cache_hash = sha256_json(
            {
                "parent": pre_cache_hash,
                "event_id": receipt.event_id,
                "weights": post_weights,
            }
        )
        return {
            "write_admitted": exact_admission.get("admitted") is True,
            "pre_cache_hash": pre_cache_hash,
            "post_cache_hash": post_cache_hash if exact_admission.get("admitted") is True else pre_cache_hash,
            "update_sign": sign,
            "magnitude": magnitude,
            "model_confidence_direction_authority": False,
            "exact_outcome_direction_authority": True,
            "touched_features": list(receipt.selected_features),
        }

    def _commit_cache_write(
        self,
        receipt: FactorCacheEventReceipt,
        cache_write: Mapping[str, Any],
    ) -> None:
        for feature in receipt.selected_features:
            current = float(self._cache.get(str(feature), {}).get("weight", 0.0))
            next_weight = current + int(cache_write["update_sign"]) * float(cache_write["magnitude"])
            self._cache[str(feature)] = {
                "factor_id": receipt.factor_id,
                "feature": str(feature),
                "weight": round(max(-WEIGHT_CAP, min(WEIGHT_CAP, next_weight)), 9),
                "event_id": receipt.event_id,
                "exact_receipt_hash": receipt.exact_receipt_hash,
            }
        self._state_hash = str(cache_write["post_cache_hash"])
        self._event_cache_hashes[receipt.event_id] = self._state_hash
        self._active_cache_hashes.add(self._state_hash)

    def _quarantine_receipt(
        self,
        receipt: FactorCacheEventReceipt,
        exact_admission: Mapping[str, Any],
    ) -> None:
        self._quarantine.append(
            {
                "event_id": receipt.event_id,
                "raw_hash": receipt.raw_hash,
                "reject_reason": exact_admission.get("reject_reason", ""),
                "quarantine_hash": sha256_json(
                    {
                        "event_id": receipt.event_id,
                        "raw_hash": receipt.raw_hash,
                        "reason": exact_admission.get("reject_reason", ""),
                    }
                ),
            }
        )

    @staticmethod
    def _bounded_magnitude(confidence: float) -> float:
        value = max(0.0, min(0.99, float(confidence)))
        return round(min(MAX_UPDATE_MAGNITUDE, value * LEARNING_RATE), 9)

    def _base_row(self, *, action: str, event_id: str, **extra: Any) -> dict[str, Any]:
        row = {
            "schema": LEDGER_SCHEMA,
            "adapter_api_version": ADAPTER_API_VERSION,
            "adapter_api_schema_hash": adapter_api_schema_hash(),
            "sequence": self._sequence,
            "action": action,
            "event_id": event_id,
            "previous_row_hash": self._ledger_tail_hash,
            **extra,
            "row_hash": "",
        }
        row["row_hash"] = _row_hash(row)
        return row

    def _append_row(self, row: Mapping[str, Any]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(dict(row)) + "\n")
        stored = dict(row)
        self._rows.append(stored)
        self._sequence += 1
        self._ledger_tail_hash = str(stored["row_hash"])

    def _restore_checkpoint(self, checkpoint_path: Path) -> None:
        parsed = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("checkpoint is not an object")
        if parsed.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint schema mismatch")
        if parsed.get("checkpoint_hash") != _checkpoint_hash(parsed):
            raise ValueError("checkpoint hash mismatch")
        if parsed.get("ledger_tail_hash") != self._ledger_tail_hash:
            raise ValueError("checkpoint ledger tail mismatch")
        self._cache = {str(key): dict(value) for key, value in parsed.get("cache", {}).items()}
        self._tombstones = {
            str(key): dict(value) for key, value in parsed.get("tombstones", {}).items()
        }
        self._quarantine = [dict(row) for row in parsed.get("quarantine", [])]
        self._rollbacks = [dict(row) for row in parsed.get("rollbacks", [])]
        self._seen_event_ids = {str(value) for value in parsed.get("seen_event_ids", [])}
        self._seen_raw_hashes = {str(value) for value in parsed.get("seen_raw_hashes", [])}
        self._event_cache_hashes = {
            str(key): str(value) for key, value in parsed.get("event_cache_hashes", {}).items()
        }
        self._active_cache_hashes = {str(value) for value in parsed.get("active_cache_hashes", [])}
        self._state_hash = str(parsed.get("state_hash", GENESIS_HASH))
        self._last_chronology_index = int(parsed.get("last_chronology_index", -1))

    def _restore_from_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            action = row.get("action")
            if action == "observe":
                event_id = str(row["event_id"])
                self._seen_event_ids.add(event_id)
                self._seen_raw_hashes.add(str(row["raw_hash"]))
                self._last_chronology_index = max(
                    self._last_chronology_index,
                    int(row.get("chronology_index", -1)),
                )
                if row.get("cache_write", {}).get("write_admitted") is True:
                    cache_hash = str(row["cache_write"]["post_cache_hash"])
                    self._event_cache_hashes[event_id] = cache_hash
                    self._active_cache_hashes.add(cache_hash)
                    self._state_hash = cache_hash
            elif action == "tombstone":
                tombstone = dict(row["tombstone"])
                self._tombstones[str(tombstone["event_id"])] = tombstone
                self._active_cache_hashes.discard(str(tombstone.get("rejected_cache_hash", "")))
            elif action == "rollback":
                rollback = dict(row["rollback"])
                self._rollbacks.append(rollback)
                self._state_hash = str(rollback["state_hash_after"])
