"""Tests for the production Safety-Net adapter.

Spec refs: REQ-PIPELINE-6549, SCENARIO-PIPELINE-6549-DEFAULT-OFF,
SCENARIO-PIPELINE-6549-ENABLED-FALLBACK, SCENARIO-PIPELINE-6549-ATTACKS.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.production_safety_net_adapter import (
    FROZEN_V566_FEATURE_NAMES,
    SafetyNetCandidate,
    SafetyNetProductionAdapter,
    SafetyNetRouterConfig,
    SafetyNetRouteRequest,
    frozen_v566_router_contract_hash,
)
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline


class _StaticExtractor:
    def __init__(self, constraints: Sequence[ConstraintResult]) -> None:
        self.constraints = list(constraints)

    def extract(self, text: str, domain: str | None = None) -> list[ConstraintResult]:
        return list(self.constraints)


class _NoopSemantic:
    def verify(self, *args: Any, **kwargs: Any) -> None:
        return None


class _TrackingPipeline(VerifyRepairPipeline):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.evaluate_orders: list[list[str]] = []
        super().__init__(*args, **kwargs)

    def _evaluate_constraints(self, constraints: list[ConstraintResult]) -> VerificationResult:
        self.evaluate_orders.append([item.description for item in constraints])
        return super()._evaluate_constraints(constraints)


def _constraint(name: str, *, satisfied: bool = True) -> ConstraintResult:
    return ConstraintResult(
        constraint_type="unit",
        description=name,
        metadata={"satisfied": satisfied, "candidate_id": name},
    )


def _public_result(result: VerificationResult) -> dict[str, object]:
    certificate = {
        key: value
        for key, value in result.certificate.items()
        if key != "production_safety_net_adapter"
    }
    return {
        "verified": result.verified,
        "energy": result.energy,
        "violations": [violation.description for violation in result.violations],
        "mode": result.mode,
        "skipped": result.skipped,
        "certificate": certificate,
    }


def _pipeline(
    *,
    constraints: Sequence[ConstraintResult],
    config: SafetyNetRouterConfig | None = None,
    ledger_path: Path | None = None,
) -> _TrackingPipeline:
    kwargs: dict[str, object] = {}
    if config is not None:
        kwargs["production_safety_net_adapter_config"] = config
    if ledger_path is not None:
        kwargs["production_safety_net_adapter_ledger_path"] = ledger_path
    return _TrackingPipeline(
        extractor=_StaticExtractor(constraints),
        semantic_grounding_verifier=_NoopSemantic(),
        semantic_verifier_v2=_NoopSemantic(),
        and_compose_verifier=False,
        **kwargs,
    )


def test_req_pipeline_6549_config_is_typed_default_off_and_contract_pinned() -> None:
    """REQ-PIPELINE-6549: typed config is default-off and hashes the V566 contract."""

    config = SafetyNetRouterConfig()
    adapter = SafetyNetProductionAdapter(config)

    assert config.enabled is False
    assert config.feature_names == FROZEN_V566_FEATURE_NAMES
    assert config.router_contract_hash == frozen_v566_router_contract_hash()
    assert (
        adapter.route(
            SafetyNetRouteRequest(
                request_id="disabled",
                candidates=(
                    SafetyNetCandidate(candidate_id="a", payload_hash="sha256:" + "1" * 64),
                ),
                feature_values={"candidate_count": 1},
            )
        )
        is None
    )
    assert adapter.adapter_configuration_contract()["enabled"] is False
    assert adapter.adapter_configuration_contract()["feature_names"] == list(
        FROZEN_V566_FEATURE_NAMES
    )


def test_scenario_pipeline_6549_disabled_pipeline_is_byte_identical(tmp_path: Path) -> None:
    """SCENARIO-PIPELINE-6549-DEFAULT-OFF: disabled adapter has no side effects."""

    constraints = [_constraint("candidate-a"), _constraint("candidate-b")]
    kwargs = {
        "question": "Pick the safe candidate.",
        "response": "candidate-a candidate-b",
        "domain": "logic",
    }
    native = _pipeline(constraints=constraints).verify(**kwargs)
    disabled = _pipeline(
        constraints=constraints,
        config=SafetyNetRouterConfig(enabled=False),
        ledger_path=tmp_path / "disabled.jsonl",
    ).verify(**kwargs)

    assert _public_result(disabled) == _public_result(native)
    assert disabled.certificate == native.certificate
    assert "production_safety_net_adapter" not in disabled.certificate
    assert disabled.violations == native.violations
    assert disabled.constraints == native.constraints
    assert not (tmp_path / "disabled.jsonl").exists()


def test_scenario_pipeline_6549_enabled_routing_preserves_exact_outputs(
    tmp_path: Path,
) -> None:
    """SCENARIO-PIPELINE-6549-ENABLED-FALLBACK: route may reorder but keeps exact output."""

    constraints = [_constraint("candidate-a"), _constraint("candidate-b")]
    kwargs = {
        "question": "Pick the safe candidate.",
        "response": "candidate-a candidate-b",
        "domain": "logic",
    }
    native_pipeline = _pipeline(constraints=constraints)
    native = native_pipeline.verify(**kwargs)
    enabled_pipeline = _pipeline(
        constraints=constraints,
        config=SafetyNetRouterConfig(enabled=True),
        ledger_path=tmp_path / "enabled.jsonl",
    )
    enabled = enabled_pipeline.verify(**kwargs)

    assert _public_result(enabled) == _public_result(native)
    assert native_pipeline.evaluate_orders == [["candidate-a", "candidate-b"]]
    assert enabled_pipeline.evaluate_orders == [["candidate-b", "candidate-a"]]
    certificate = enabled.certificate["production_safety_net_adapter"]
    assert certificate["route"] == "compact_router"
    assert certificate["abstention"] is False
    assert certificate["fallback_reason"] == ""
    assert certificate["candidate_preservation"]["all_candidates_preserved"] is True
    assert certificate["exact_result"]["verified"] is True
    assert certificate["charged_adapter_overhead_units"] > 0.0
    assert Path(tmp_path / "enabled.jsonl").exists()


def test_scenario_pipeline_6549_abstain_exception_malformed_and_rollback(
    tmp_path: Path,
) -> None:
    """SCENARIO-PIPELINE-6549-ATTACKS: uncertain and attacked inputs fail closed."""

    exception_key = SafetyNetProductionAdapter.exception_key(
        candidate_ids=("candidate-a",),
        split_name="train",
    )
    adapter = SafetyNetProductionAdapter(
        SafetyNetRouterConfig(
            enabled=True,
            exception_table={exception_key: "native_exact_fallback"},
        )
    )
    table_decision = adapter.route(
        SafetyNetRouteRequest.from_candidate_ids(
            request_id="table",
            candidate_ids=("candidate-a",),
            split_name="train",
        )
    )
    assert table_decision is not None
    assert table_decision.exception_lookup["hit"] is True
    assert table_decision.fallback_reason == "exception_table_hit"
    assert table_decision.chosen_order == ("candidate-a",)

    abstain_decision = SafetyNetProductionAdapter(SafetyNetRouterConfig(enabled=True)).route(
        SafetyNetRouteRequest.from_candidate_ids(
            request_id="abstain",
            candidate_ids=("candidate-a",),
        )
    )
    assert abstain_decision is not None
    assert abstain_decision.abstention is True
    assert abstain_decision.fallback_reason == "abstention"

    stale = SafetyNetProductionAdapter(
        SafetyNetRouterConfig(enabled=True, router_contract_hash="sha256:" + "f" * 64)
    ).route(
        SafetyNetRouteRequest.from_candidate_ids(
            request_id="stale",
            candidate_ids=("candidate-a", "candidate-b"),
        )
    )
    assert stale is not None
    assert stale.route == "native_exact_fallback"
    assert stale.fallback_reason == "stale_configuration"

    duplicate = SafetyNetProductionAdapter(SafetyNetRouterConfig(enabled=True)).route(
        SafetyNetRouteRequest.from_candidate_ids(
            request_id="duplicate",
            candidate_ids=("candidate-a", "candidate-a"),
        )
    )
    assert duplicate is not None
    assert duplicate.route == "native_exact_fallback"
    assert duplicate.fallback_reason == "malformed_input:duplicate_candidate_ids"

    rollback_adapter = SafetyNetProductionAdapter(
        SafetyNetRouterConfig(enabled=True), ledger_path=tmp_path / "rollback.jsonl"
    )
    rollback_adapter.rollback("operator_disable")
    assert rollback_adapter.config.enabled is False
    assert (
        rollback_adapter.route(
            SafetyNetRouteRequest.from_candidate_ids(
                request_id="rolled-back",
                candidate_ids=("candidate-a", "candidate-b"),
            )
        )
        is None
    )


def test_scenario_pipeline_6549_private_fail_closed_edges(tmp_path: Path) -> None:
    """SCENARIO-PIPELINE-6549-ATTACKS: private validation branches close safely."""

    good = SafetyNetRouteRequest.from_candidate_ids(
        request_id="good",
        candidate_ids=("candidate-a", "candidate-b"),
    )
    feature_stale = SafetyNetProductionAdapter(
        SafetyNetRouterConfig(enabled=True, feature_names=("candidate_count",))
    ).route(good)
    family_stale = SafetyNetProductionAdapter(
        SafetyNetRouterConfig(enabled=True, model_family="unknown")
    ).route(good)
    assert feature_stale is not None
    assert feature_stale.fallback_reason == "stale_configuration"
    assert family_stale is not None
    assert family_stale.fallback_reason == "stale_configuration"

    empty = SafetyNetProductionAdapter(SafetyNetRouterConfig(enabled=True)).route(
        SafetyNetRouteRequest(request_id="empty", candidates=())
    )
    blank = SafetyNetProductionAdapter(SafetyNetRouterConfig(enabled=True)).route(
        SafetyNetRouteRequest.from_candidate_ids(
            request_id="blank",
            candidate_ids=("",),
        )
    )
    bad_hash = SafetyNetProductionAdapter(SafetyNetRouterConfig(enabled=True)).route(
        SafetyNetRouteRequest(
            request_id="bad-hash",
            candidates=(SafetyNetCandidate(candidate_id="a", payload_hash="bad"),),
            feature_values={"candidate_count": 1},
        )
    )
    forbidden = SafetyNetProductionAdapter(SafetyNetRouterConfig(enabled=True)).route(
        SafetyNetRouteRequest(
            request_id="forbidden",
            candidates=(SafetyNetCandidate(candidate_id="a", payload_hash="sha256:" + "1" * 64),),
            feature_values={"source_id": 1},
        )
    )
    unsupported = SafetyNetProductionAdapter(SafetyNetRouterConfig(enabled=True)).route(
        SafetyNetRouteRequest(
            request_id="unsupported",
            candidates=(SafetyNetCandidate(candidate_id="a", payload_hash="sha256:" + "1" * 64),),
            feature_values={"unsupported": 1},
        )
    )

    assert empty is not None
    assert empty.fallback_reason == "malformed_input:no_candidates"
    assert blank is not None
    assert blank.fallback_reason == "malformed_input:blank_candidate_id"
    assert bad_hash is not None
    assert bad_hash.fallback_reason == "malformed_input:bad_payload_hash"
    assert forbidden is not None
    assert forbidden.fallback_reason == "malformed_input:forbidden_feature"
    assert unsupported is not None
    assert unsupported.fallback_reason == "malformed_input:unsupported_feature"

    decision = SafetyNetProductionAdapter(SafetyNetRouterConfig(enabled=True)).route(good)
    assert decision is not None
    assert "decision_hash" in decision.to_dict()
    no_ledger = SafetyNetProductionAdapter(SafetyNetRouterConfig(enabled=True))
    no_ledger.record_exact_result(decision, {"verified": True})
    no_ledger._write_row({})  # noqa: SLF001
    assert not list(tmp_path.iterdir())
