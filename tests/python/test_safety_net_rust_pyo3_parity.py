"""Tests for the Safety-Net Rust/PyO3 routing ABI.

Spec refs: REQ-RUSTPY-6550, REQ-RUSTPY-6550-SCHEMA,
REQ-RUSTPY-6550-NUMERIC, REQ-RUSTPY-6550-SERIALIZATION,
REQ-RUSTPY-6550-ERRORS, REQ-RUSTPY-6550-PARITY,
REQ-RUSTPY-6550-FALLBACK, SCENARIO-RUSTPY-6550-BOUNDARY-PARITY,
REQ-RUSTPY-6564, REQ-RUSTPY-6564-BATCH-SCHEMA,
REQ-RUSTPY-6564-BATCH-PARITY, REQ-RUSTPY-6564-BATCH-ERRORS,
SCENARIO-RUSTPY-6564-BATCH-ORDERED-PARITY.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline import safety_net_abi as abi


rust = pytest.importorskip("carnot._rust")


def _rust_route(payload: dict[str, object]) -> dict[str, object]:
    request_bytes = abi.canonical_request_bytes(payload)
    return dict(rust.safety_net_route_bytes(request_bytes))


def test_req_rustpy_6550_spec_declares_router_abi_contract() -> None:
    """REQ-RUSTPY-6550: OpenSpec owns the Safety-Net ABI contract."""

    text = Path("openspec/capabilities/rust-python-boundary/spec.md").read_text(encoding="utf-8")
    section = text[text.index("REQ-RUSTPY-6550") :]
    for marker in (
        "REQ-RUSTPY-6550-SCHEMA",
        "REQ-RUSTPY-6550-NUMERIC",
        "REQ-RUSTPY-6550-SERIALIZATION",
        "REQ-RUSTPY-6550-ERRORS",
        "REQ-RUSTPY-6550-PARITY",
        "REQ-RUSTPY-6550-FALLBACK",
        "REQ-RUSTPY-6550-ROLLBACK",
        "REQ-RUSTPY-6550-NO-AUTHORITY",
        "SCENARIO-RUSTPY-6550-BOUNDARY-PARITY",
    ):
        assert marker in section


def test_req_rustpy_6564_spec_declares_batch_router_abi_contract() -> None:
    """REQ-RUSTPY-6564: OpenSpec owns the Safety-Net batch ABI contract."""

    text = Path("openspec/capabilities/rust-python-boundary/spec.md").read_text(encoding="utf-8")
    section = text[text.index("REQ-RUSTPY-6564") :]
    for marker in (
        "REQ-RUSTPY-6564-BATCH-SCHEMA",
        "REQ-RUSTPY-6564-BATCH-PARITY",
        "REQ-RUSTPY-6564-BATCH-ERRORS",
        "REQ-RUSTPY-6564-NO-SCOPE-CREEP",
        "SCENARIO-RUSTPY-6564-BATCH-ORDERED-PARITY",
    ):
        assert marker in section


def test_req_rustpy_6550_supported_route_and_exception_parity() -> None:
    """REQ-RUSTPY-6550-PARITY: Python and Rust route supported rows equally."""

    payload = abi.request_payload(
        request_id="held-route",
        candidate_ids=("sha256:" + "1" * 64, "sha256:" + "2" * 64),
        feature_values={"candidate_count": 2.0, "constraint_count": 2},
    )
    exception_key = abi.exception_key(
        candidate_ids=payload["candidate_ids"],
        split_name="train",
    )
    exception_payload = abi.request_payload(
        request_id="train-exception",
        candidate_ids=tuple(payload["candidate_ids"]),
        split_name="train",
        exception_table={exception_key: "native_exact_fallback"},
    )

    for item in (payload, exception_payload):
        request_bytes = abi.canonical_request_bytes(item)
        py_decision = abi.route_request_bytes(request_bytes)
        rust_decision = dict(rust.safety_net_route_bytes(request_bytes))
        assert rust_decision == py_decision
        assert rust_decision["schema_version"] == abi.ABI_SCHEMA_VERSION
        assert abi.canonical_decision_bytes(rust_decision) == abi.canonical_decision_bytes(
            py_decision
        )

    assert _rust_route(payload)["route"] == "compact_router"
    assert _rust_route(exception_payload)["exception_hit"] is True
    assert _rust_route(exception_payload)["fallback_reason"] == "exception_table_hit"


def test_scenario_rustpy_6564_batch_route_preserves_scalar_order_and_bytes() -> None:
    """SCENARIO-RUSTPY-6564-BATCH-ORDERED-PARITY: batch equals scalar routing."""

    c1 = "sha256:" + "1" * 64
    c2 = "sha256:" + "2" * 64
    exception = abi.exception_key(candidate_ids=(c1, c2), split_name="train")
    payloads = (
        abi.request_payload(
            request_id="batch-supported",
            candidate_ids=(c1, c2),
            feature_values={"candidate_count": 2.0, "constraint_count": 2},
        ),
        abi.request_payload(request_id="batch-abstain", candidate_ids=(c1,)),
        abi.request_payload(
            request_id="batch-forced",
            candidate_ids=(c1, c2),
            forced_fallback_reason="forced_fallback",
        ),
        abi.request_payload(
            request_id="batch-exception",
            candidate_ids=(c1, c2),
            split_name="train",
            exception_table={exception: "native_exact_fallback"},
        ),
        abi.request_payload(
            request_id="batch-unsupported",
            candidate_ids=(c1, c2),
            router_contract_hash="sha256:" + "f" * 64,
        ),
        abi.request_payload(
            request_id="batch-malformed",
            candidate_ids=(c1,),
            feature_values={"unknown": 1},
        ),
    )
    request_bytes = [abi.canonical_request_bytes(payload) for payload in payloads]
    request_bytes.append(abi.nan_attack_request_bytes())

    scalar = [dict(rust.safety_net_route_bytes(item)) for item in request_bytes]
    batch = [dict(item) for item in rust.safety_net_route_batch(request_bytes)]
    python = [abi.route_request_bytes(item) for item in request_bytes]

    assert len(batch) == len(request_bytes)
    assert batch == scalar == python
    assert [row["request_hash"] for row in batch] == [
        abi.route_request_bytes(item)["request_hash"] for item in request_bytes
    ]
    assert [abi.canonical_decision_bytes(row) for row in batch] == [
        abi.canonical_decision_bytes(row) for row in scalar
    ]
    assert any(row["route"] == "compact_router" for row in batch)
    assert any(row["fallback_reason"] == "abstention" for row in batch)
    assert any(row["fallback_reason"] == "forced_fallback" for row in batch)
    assert any(row["fallback_reason"] == "exception_table_hit" for row in batch)
    assert any(row["fallback_reason"] == "stale_configuration" for row in batch)
    assert any(row["error_type"] == "JsonDecodeError" for row in batch)


def test_req_rustpy_6550_typed_pyo3_request_and_decision_surface() -> None:
    """REQ-RUSTPY-6550-SCHEMA: PyO3 exposes typed request and decision objects."""

    payload = abi.request_payload(
        request_id="typed",
        candidate_ids=("sha256:" + "3" * 64, "sha256:" + "4" * 64),
    )
    typed_request = rust.RustSafetyNetFeatureRequest(abi.canonical_request_bytes(payload))
    typed_decision = typed_request.decision()

    assert typed_request.input_hash().startswith("sha256:")
    assert typed_decision.route == "compact_router"
    assert typed_decision.abstain is False
    assert typed_decision.uncertainty_bucket == "medium"
    assert typed_decision.exception_hit is False
    assert typed_decision.fallback_reason == ""
    assert typed_decision.schema_version == abi.ABI_SCHEMA_VERSION
    assert typed_decision.to_dict() == abi.route_request_bytes(abi.canonical_request_bytes(payload))
    assert json.loads(typed_decision.canonical_json()) == typed_decision.to_dict()


def test_req_rustpy_6550_errors_and_numeric_attacks_fail_closed() -> None:
    """REQ-RUSTPY-6550-ERRORS: malformed requests fail closed the same way."""

    cases = (
        abi.request_payload(request_id="single", candidate_ids=("sha256:" + "5" * 64,)),
        {**abi.request_payload(request_id="missing", candidate_ids=()), "candidate_ids": None},
        abi.request_payload(
            request_id="unknown-feature",
            candidate_ids=("sha256:" + "6" * 64,),
            feature_values={"unknown": 1},
        ),
        abi.request_payload(
            request_id="extra",
            candidate_ids=("sha256:" + "7" * 64,),
            extra={"source_id": "forbidden"},
        ),
        abi.request_payload(
            request_id="extreme",
            candidate_ids=("sha256:" + "8" * 64,),
            feature_values={"candidate_count": 10**15},
        ),
        abi.request_payload(
            request_id="unicode-candidate",
            candidate_ids=("candidate-micro-\u00b5",),
        ),
        abi.request_payload(
            request_id="stale",
            candidate_ids=("sha256:" + "9" * 64,),
            schema_version="carnot.safety_net.router_abi.v0",
        ),
    )
    for payload in cases:
        request_bytes = abi.canonical_request_bytes(payload)
        py_decision = abi.route_request_bytes(request_bytes)
        rust_decision = dict(rust.safety_net_route_bytes(request_bytes))
        assert rust_decision == py_decision
        assert rust_decision["route"] == "native_exact_fallback"
        assert rust_decision["fallback_reason"]
        assert rust_decision["exact_fallback_reachable"] is True

    nan_decision = dict(rust.safety_net_route_bytes(abi.nan_attack_request_bytes()))
    assert nan_decision == abi.route_request_bytes(abi.nan_attack_request_bytes())
    assert nan_decision["fallback_reason"] == "malformed_input:invalid_json"
    assert nan_decision["error_type"] == "JsonDecodeError"


def test_req_rustpy_6550_serialization_normalizes_order_and_integer_floats() -> None:
    """REQ-RUSTPY-6550-SERIALIZATION: canonical bytes remove map-order drift."""

    base = abi.request_payload(
        request_id="ordered",
        candidate_ids=("sha256:" + "a" * 64, "sha256:" + "b" * 64),
        feature_values={"constraint_count": 2, "candidate_count": 2.0},
    )
    reordered = {
        "seed": base["seed"],
        "feature_values": {"candidate_count": 2, "constraint_count": 2.0},
        "candidate_ids": base["candidate_ids"],
        "request_id": base["request_id"],
        "schema_version": base["schema_version"],
        "split_name": base["split_name"],
        "router_contract_hash": base["router_contract_hash"],
        "exception_table": base["exception_table"],
        "forced_abstain": base["forced_abstain"],
        "forced_fallback_reason": base["forced_fallback_reason"],
    }

    assert abi.canonical_request_bytes(base) == abi.canonical_request_bytes(reordered)
    assert abi.route_request_bytes(abi.canonical_request_bytes(base)) == dict(
        rust.safety_net_route_bytes(abi.canonical_request_bytes(reordered))
    )


def test_req_rustpy_6550_python_defensive_error_branches() -> None:
    """REQ-RUSTPY-6550-ERRORS: Python ABI covers each explicit error branch."""

    assert abi.route_request_bytes(b"null")["fallback_reason"] == "malformed_input:not_object"

    stale_config = abi.request_payload(
        request_id="stale-config",
        candidate_ids=("sha256:" + "c" * 64, "sha256:" + "d" * 64),
        router_contract_hash="sha256:" + "f" * 64,
    )
    compact = abi.route_request(
        abi.request_payload(
            request_id="direct-compact",
            candidate_ids=("sha256:" + "e" * 64, "sha256:" + "f" * 64),
        )
    )
    assert compact["route"] == "compact_router"
    assert abi.route_request(stale_config)["fallback_reason"] == "stale_configuration"

    malformed_payloads = (
        (
            {
                **abi.request_payload(request_id="features", candidate_ids=("x",)),
                "feature_values": [],
            },
            "malformed_input:feature_values_not_object",
        ),
        (
            abi.request_payload(
                request_id="forbidden-feature",
                candidate_ids=("x",),
                feature_values={"source_id": 1},
            ),
            "malformed_input:forbidden_feature",
        ),
        (
            {
                **abi.request_payload(request_id="exceptions", candidate_ids=("x",)),
                "exception_table": [],
            },
            "malformed_input:exception_table_not_object",
        ),
        (
            {
                **abi.request_payload(request_id="abstain-type", candidate_ids=("x",)),
                "forced_abstain": "yes",
            },
            "malformed_input:forced_abstain_not_bool",
        ),
        (
            {
                **abi.request_payload(request_id="fallback-type", candidate_ids=("x",)),
                "forced_fallback_reason": 1,
            },
            "malformed_input:forced_fallback_not_string",
        ),
        (
            abi.request_payload(request_id="empty", candidate_ids=()),
            "malformed_input:no_candidates",
        ),
        (
            abi.request_payload(request_id="blank", candidate_ids=("",)),
            "malformed_input:blank_candidate_id",
        ),
        (
            abi.request_payload(
                request_id="non-numeric",
                candidate_ids=("x",),
                feature_values={"candidate_count": "two"},
            ),
            "malformed_input:non_numeric_feature",
        ),
        (
            abi.request_payload(
                request_id="non-finite",
                candidate_ids=("x",),
                feature_values={"candidate_count": float("inf")},
            ),
            "malformed_input:non_finite_feature",
        ),
        (
            abi.request_payload(
                request_id="non-integer",
                candidate_ids=("x",),
                feature_values={"candidate_count": 1.25},
            ),
            "malformed_input:non_integer_feature",
        ),
    )
    for payload, expected in malformed_payloads:
        assert abi.route_request(payload)["fallback_reason"] == expected
