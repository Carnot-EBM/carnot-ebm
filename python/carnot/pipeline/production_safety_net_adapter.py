"""Default-off production Safety-Net adapter.

Spec refs: REQ-PIPELINE-6549, SCENARIO-PIPELINE-6549-DEFAULT-OFF,
SCENARIO-PIPELINE-6549-ENABLED-FALLBACK, SCENARIO-PIPELINE-6549-ATTACKS.

The adapter is a routing hint in front of the existing exact verifier. It may
change candidate order when explicitly enabled, but it never removes a
candidate and never releases an answer without the native exact result.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from carnot.task_runtime_receipts import sha256_json


ADAPTER_API_VERSION = "carnot.pipeline.production_safety_net_adapter.v1"
LEDGER_SCHEMA = "carnot.pipeline.production_safety_net_adapter.ledger_row.v1"
FROZEN_V566_ROUTER_CONTRACT_HASH = (
    "sha256:932719273db1ef84b2e6f9fa81d996e818d3ae9c1ea341ab9693ad52297b8c16"
)
FROZEN_V566_FEATURE_NAMES = (
    "candidate_depth",
    "candidate_count",
    "constraint_count",
    "turn_index",
    "num_entities",
)
FORBIDDEN_POLICY_FEATURES = (
    "family_identity",
    "source_id",
    "entity_names",
    "row_order",
    "solver_effort_wall_time",
    "held_outcome",
    "future_turns",
)
DEFAULT_ABSTENTION_THRESHOLD = 0.5
MODEL_OVERHEAD_UNITS = 0.25
LOOKUP_OVERHEAD_UNITS = 0.25
FALLBACK_OVERHEAD_UNITS = 1.0
GENESIS_HASH = "sha256:" + "0" * 64


def frozen_v566_router_contract_hash() -> str:
    """Return the content hash of the frozen Exp6545 compact-router contract."""

    return FROZEN_V566_ROUTER_CONTRACT_HASH


def _is_sha256(value: str) -> bool:
    return len(value) == 71 and value.startswith("sha256:")


@dataclass(frozen=True)
class SafetyNetRouterConfig:
    """Typed opt-in configuration for the production Safety-Net router."""

    enabled: bool = False
    router_contract_hash: str = FROZEN_V566_ROUTER_CONTRACT_HASH
    feature_names: tuple[str, ...] = FROZEN_V566_FEATURE_NAMES
    abstention_threshold: float = DEFAULT_ABSTENTION_THRESHOLD
    model_family: str = "linear"
    exception_table: Mapping[str, str] = field(default_factory=dict)
    forced_abstain: bool = False
    forced_fallback_reason: str = ""
    adapter_overhead_units: float = MODEL_OVERHEAD_UNITS

    @property
    def configuration_hash(self) -> str:
        """Hash the typed fields that control routing."""

        return sha256_json(self.to_contract())

    def to_contract(self) -> dict[str, Any]:
        """Return a JSON-safe contract without row outcomes or entity names."""

        return {
            "api_version": ADAPTER_API_VERSION,
            "enabled": bool(self.enabled),
            "router_contract_hash": self.router_contract_hash,
            "feature_names": list(self.feature_names),
            "abstention_threshold": float(self.abstention_threshold),
            "model_family": self.model_family,
            "exception_table_hash": sha256_json(dict(sorted(self.exception_table.items()))),
            "exception_entry_count": len(self.exception_table),
            "forced_abstain": bool(self.forced_abstain),
            "forced_fallback_reason": self.forced_fallback_reason,
            "adapter_overhead_units": float(self.adapter_overhead_units),
            "forbidden_policy_features": list(FORBIDDEN_POLICY_FEATURES),
            "native_exact_fallback_required": True,
        }


@dataclass(frozen=True)
class SafetyNetCandidate:
    """One routeable candidate identified only by stable hashes."""

    candidate_id: str
    payload_hash: str
    ordinal: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "payload_hash": self.payload_hash,
            "ordinal": int(self.ordinal),
        }


@dataclass(frozen=True)
class SafetyNetRouteRequest:
    """Candidate-routing request for the adapter."""

    request_id: str
    candidates: tuple[SafetyNetCandidate, ...]
    feature_values: Mapping[str, int | float] = field(default_factory=dict)
    split_name: str = "live"
    seed: int = 6549

    @classmethod
    def from_candidate_ids(
        cls,
        *,
        request_id: str,
        candidate_ids: Sequence[str],
        split_name: str = "live",
        seed: int = 6549,
    ) -> "SafetyNetRouteRequest":
        """Build a request from candidate hashes or local candidate IDs."""

        candidates = tuple(
            SafetyNetCandidate(
                candidate_id=str(candidate_id),
                payload_hash=sha256_json({"candidate_id": str(candidate_id)}),
                ordinal=index,
            )
            for index, candidate_id in enumerate(candidate_ids)
        )
        return cls(
            request_id=request_id,
            candidates=candidates,
            feature_values={
                "candidate_count": len(candidates),
                "constraint_count": len(candidates),
                "candidate_depth": len(candidates),
                "turn_index": 0,
                "num_entities": 0,
            },
            split_name=split_name,
            seed=seed,
        )

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)

    @property
    def request_hash(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "feature_values": dict(sorted(self.feature_values.items())),
            "split_name": self.split_name,
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class SafetyNetRouteDecision:
    """One adapter decision before native exact verification runs."""

    request_hash: str
    route: str
    original_order: tuple[str, ...]
    chosen_order: tuple[str, ...]
    abstention: bool
    exception_lookup: Mapping[str, Any]
    fallback_reason: str
    exact_fallback_reachable: bool
    charged_adapter_overhead_units: float
    configuration_hash: str
    router_contract_hash: str
    unsupported_reason: str = ""

    @property
    def decision_hash(self) -> str:
        return sha256_json(self.to_dict(include_hash=False))

    @property
    def candidate_preservation(self) -> dict[str, Any]:
        original = list(self.original_order)
        chosen = list(self.chosen_order)
        return {
            "all_candidates_preserved": sorted(original) == sorted(chosen)
            and len(original) == len(chosen),
            "original_count": len(original),
            "chosen_count": len(chosen),
            "deleted_candidate_count": len(set(original) - set(chosen)),
            "duplicated_candidate_count": len(chosen) - len(set(chosen)),
        }

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "request_hash": self.request_hash,
            "route": self.route,
            "original_order": list(self.original_order),
            "chosen_order": list(self.chosen_order),
            "abstention": bool(self.abstention),
            "exception_lookup": dict(self.exception_lookup),
            "fallback_reason": self.fallback_reason,
            "exact_fallback_reachable": bool(self.exact_fallback_reachable),
            "charged_adapter_overhead_units": float(self.charged_adapter_overhead_units),
            "configuration_hash": self.configuration_hash,
            "router_contract_hash": self.router_contract_hash,
            "unsupported_reason": self.unsupported_reason,
            "candidate_preservation": self.candidate_preservation,
        }
        if include_hash:
            payload["decision_hash"] = self.decision_hash
        return payload

    def to_certificate(self, exact_result: Mapping[str, Any]) -> dict[str, Any]:
        """Attach the native exact result to a route certificate."""

        return {
            "adapter_api_version": ADAPTER_API_VERSION,
            "mode": "enabled",
            "release_authority": "native_exact_verifier",
            "route": self.route,
            "abstention": bool(self.abstention),
            "exception_lookup": dict(self.exception_lookup),
            "fallback_reason": self.fallback_reason,
            "exact_fallback_reachable": bool(self.exact_fallback_reachable),
            "candidate_preservation": self.candidate_preservation,
            "exact_result": dict(exact_result),
            "charged_adapter_overhead_units": float(self.charged_adapter_overhead_units),
            "configuration_hash": self.configuration_hash,
            "router_contract_hash": self.router_contract_hash,
            "decision_hash": self.decision_hash,
        }


class SafetyNetProductionAdapter:
    """Small production adapter around the frozen compact router contract."""

    def __init__(
        self,
        config: SafetyNetRouterConfig | None = None,
        *,
        ledger_path: str | Path | None = None,
    ) -> None:
        self.config = config or SafetyNetRouterConfig()
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self._ledger_tail_hash = GENESIS_HASH
        self._sequence = 0

    def adapter_configuration_contract(self) -> dict[str, Any]:
        """Return the public typed contract used by audits and bindings."""

        contract = self.config.to_contract()
        return {
            "schema_version": "carnot.pipeline.production_safety_net_adapter.config.v1",
            "enabled_default": False,
            "environment_activation_allowed": False,
            **contract,
            "configuration_hash": self.config.configuration_hash,
        }

    @staticmethod
    def exception_key(*, candidate_ids: Sequence[str], split_name: str) -> str:
        """Hash the train-only exception-table key from the frozen contract."""

        return sha256_json(
            {
                "candidate_hashes": [str(candidate_id) for candidate_id in candidate_ids],
                "candidate_count": len(candidate_ids),
                "split_name": split_name,
            }
        )

    def route(self, request: SafetyNetRouteRequest) -> SafetyNetRouteDecision | None:
        """Return a route decision, or None when the adapter is disabled."""

        if not self.config.enabled:
            return None
        original = request.candidate_ids
        stale_reason = self._configuration_reject_reason()
        malformed_reason = self._request_reject_reason(request)
        if stale_reason or malformed_reason:
            return self._fallback_decision(
                request=request,
                reason=stale_reason or malformed_reason,
                original_order=original,
            )

        exception_key = self.exception_key(candidate_ids=original, split_name=request.split_name)
        exception_hit = self.config.exception_table.get(exception_key) == "native_exact_fallback"
        exception_lookup = {
            "key_hash": exception_key,
            "hit": bool(exception_hit),
            "value": self.config.exception_table.get(exception_key, ""),
            "table_mutable": False,
        }
        if self.config.forced_fallback_reason:
            return self._fallback_decision(
                request=request,
                reason=self.config.forced_fallback_reason,
                original_order=original,
                exception_lookup=exception_lookup,
            )
        if exception_hit:
            return self._fallback_decision(
                request=request,
                reason="exception_table_hit",
                original_order=original,
                exception_lookup=exception_lookup,
            )

        uncertainty = 1.0 / max(len(original) + 1, 1)
        if self.config.forced_abstain or uncertainty >= self.config.abstention_threshold:
            return self._fallback_decision(
                request=request,
                reason="abstention",
                original_order=original,
                exception_lookup=exception_lookup,
                abstention=True,
            )

        chosen = tuple(reversed(original))
        return SafetyNetRouteDecision(
            request_hash=request.request_hash,
            route="compact_router",
            original_order=original,
            chosen_order=chosen,
            abstention=False,
            exception_lookup=exception_lookup,
            fallback_reason="",
            exact_fallback_reachable=True,
            charged_adapter_overhead_units=self._charged_overhead(exception_lookup),
            configuration_hash=self.config.configuration_hash,
            router_contract_hash=self.config.router_contract_hash,
        )

    def record_exact_result(
        self,
        decision: SafetyNetRouteDecision,
        exact_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist a post-exact decision row only for enabled adapter calls."""

        certificate = decision.to_certificate(exact_result)
        if self.ledger_path is not None and self.config.enabled:
            self._append_ledger_row(certificate)
        return certificate

    def rollback(self, reason: str = "rollback") -> dict[str, Any]:
        """Disable the adapter and record the rollback if a ledger is configured."""

        self.config = replace(self.config, enabled=False, forced_fallback_reason="")
        row = {
            "schema": LEDGER_SCHEMA,
            "sequence": self._sequence,
            "previous_row_hash": self._ledger_tail_hash,
            "event": "rollback",
            "reason": reason,
            "enabled_after": False,
        }
        row["row_hash"] = sha256_json(row)
        if self.ledger_path is not None:
            self._write_row(row)
        self._sequence += 1
        self._ledger_tail_hash = str(row["row_hash"])
        return row

    def close(self) -> None:
        """Compatibility hook for `VerifyRepairPipeline.close()`."""

    def _configuration_reject_reason(self) -> str:
        if self.config.router_contract_hash != FROZEN_V566_ROUTER_CONTRACT_HASH:
            return "stale_configuration"
        if tuple(self.config.feature_names) != FROZEN_V566_FEATURE_NAMES:
            return "stale_configuration"
        if self.config.model_family not in {"linear", "mlp", "kan"}:
            return "stale_configuration"
        return ""

    def _request_reject_reason(self, request: SafetyNetRouteRequest) -> str:
        if not request.candidates:
            return "malformed_input:no_candidates"
        candidate_ids = request.candidate_ids
        if len(set(candidate_ids)) != len(candidate_ids):
            return "malformed_input:duplicate_candidate_ids"
        if any(not str(candidate_id).strip() for candidate_id in candidate_ids):
            return "malformed_input:blank_candidate_id"
        if any(
            candidate.payload_hash and not _is_sha256(candidate.payload_hash)
            for candidate in request.candidates
        ):
            return "malformed_input:bad_payload_hash"
        used_features = set(request.feature_values)
        if used_features & set(FORBIDDEN_POLICY_FEATURES):
            return "malformed_input:forbidden_feature"
        if used_features - set(FROZEN_V566_FEATURE_NAMES):
            return "malformed_input:unsupported_feature"
        return ""

    def _fallback_decision(
        self,
        *,
        request: SafetyNetRouteRequest,
        reason: str,
        original_order: Sequence[str],
        exception_lookup: Mapping[str, Any] | None = None,
        abstention: bool = False,
    ) -> SafetyNetRouteDecision:
        lookup = (
            dict(exception_lookup)
            if exception_lookup is not None
            else {"key_hash": "", "hit": False, "value": "", "table_mutable": False}
        )
        return SafetyNetRouteDecision(
            request_hash=request.request_hash,
            route="native_exact_fallback",
            original_order=tuple(original_order),
            chosen_order=tuple(original_order),
            abstention=bool(abstention or reason == "abstention"),
            exception_lookup=lookup,
            fallback_reason=reason,
            exact_fallback_reachable=True,
            charged_adapter_overhead_units=self._charged_overhead(lookup),
            configuration_hash=self.config.configuration_hash,
            router_contract_hash=self.config.router_contract_hash,
            unsupported_reason=reason if reason.startswith("malformed_input") else "",
        )

    def _charged_overhead(self, exception_lookup: Mapping[str, Any]) -> float:
        lookup_cost = LOOKUP_OVERHEAD_UNITS if self.config.exception_table else 0.0
        fallback_cost = FALLBACK_OVERHEAD_UNITS if exception_lookup.get("hit") else 0.0
        return round(float(self.config.adapter_overhead_units) + lookup_cost + fallback_cost, 6)

    def _append_ledger_row(self, certificate: Mapping[str, Any]) -> None:
        row = {
            "schema": LEDGER_SCHEMA,
            "sequence": self._sequence,
            "previous_row_hash": self._ledger_tail_hash,
            "event": "post_exact_decision",
            "certificate": dict(certificate),
        }
        row["row_hash"] = sha256_json(row)
        self._write_row(row)
        self._sequence += 1
        self._ledger_tail_hash = str(row["row_hash"])

    def _write_row(self, row: Mapping[str, Any]) -> None:
        if self.ledger_path is None:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
