"""Tests for the first-party ARC tool-gap receipt contract.

Spec refs: REQ-ARC-6859 and every SCENARIO-ARC-6859-* section.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.agentic.arc_induction_tools import InductionToolSession, dispatch_tool
from carnot.agentic import arc_tool_gap_receipt as receipt


class Clock:
    """Return stable increasing timestamps so fixture hashes are repeatable."""

    def __init__(self) -> None:
        self.tick = 0

    def __call__(self) -> str:
        self.tick += 1
        return f"2026-09-01T00:00:{self.tick:02d}Z"


def _transport(path: Path, *, provenance_class: str = "fixture"):
    return receipt.FirstPartyToolGapReceiptTransport(
        path,
        attempt_identity="attempt-6859",
        game_id="fixture-game",
        provenance_class=provenance_class,
        clock=Clock(),
    )


def _unknown_gap(transport: receipt.FirstPartyToolGapReceiptTransport) -> str:
    session = InductionToolSession([])
    result = dispatch_tool(
        session,
        "missing_grid_tool",
        '{"region": 3}',
        receipt_transport=transport,
        decision_point_identity="decision-1",
    )
    assert result["ok"] is False
    return transport.last_receipt_identity


def test_scenario_arc_6859_no_gap_and_default_off(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-6859-NO-GAP-AND-DEFAULT-OFF preserves action identity."""

    session = InductionToolSession([])
    transport = _transport(tmp_path / "receipts.json")
    assert (
        dispatch_tool(
            session,
            "list_transitions",
            "{}",
            receipt_transport=transport,
            decision_point_identity="decision-clean",
        )["ok"]
        is True
    )
    assert transport.snapshot()["rows"] == []

    monkeypatch.delenv(receipt.ENABLE_ENV, raising=False)
    monkeypatch.setenv(receipt.PATH_ENV, str(tmp_path / "default-off.json"))
    assert receipt.maybe_make_first_party_tool_gap_receipt_transport("g", "run") is None
    assert not (tmp_path / "default-off.json").exists()

    action = (1, None)
    policy = object.__new__(E3AgentPolicy)
    policy._first_party_tool_gap_receipt_transport = None
    policy._provenance = None
    policy._next_move_routed = lambda frames, latest: action
    policy._maybe_apply_trace_automaton_action = lambda move, latest: move
    policy.record_target_licensed_route_shadow = lambda move, **kwargs: move
    policy.record_typed_obligation_shadow_monitor = lambda move, **kwargs: move
    policy._record_outcome_transport_proposal = lambda *args: None
    assert policy.next_move([], None) is action


def test_scenario_arc_6859_detected_gap_rejected_request_and_source_hashes(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-6859-GAP-REJECTION-AND-TOOL-ERROR binds all early hops."""

    transport = _transport(tmp_path / "receipts.json")
    receipt_identity = _unknown_gap(transport)
    row = transport.snapshot()["rows"][0]

    assert row["receipt_identity"] == receipt_identity
    assert row["gap_kind"] == "unknown_tool"
    assert row["request_rejected"] is True
    assert set(row["hops"]) == {"gap_detection", "request", "tool_response"}
    assert all(hop["receipt_identity"] == receipt_identity for hop in row["hops"].values())
    assert all(hop["timestamp"].endswith("Z") for hop in row["hops"].values())
    assert all(hop["source_sha256"].startswith("sha256:") for hop in row["hops"].values())
    assert all(hop["payload_sha256"].startswith("sha256:") for hop in row["hops"].values())
    assert (
        row["hops"]["request"]["previous_hop_identity"]
        == row["hops"]["gap_detection"]["hop_identity"]
    )


def test_scenario_arc_6859_tool_error_hidden_and_delivered_response(tmp_path: Path) -> None:
    """SCENARIO-ARC-6859-VISIBILITY-USE-AND-NEXT-ACTION separates delivery."""

    transport = _transport(tmp_path / "receipts.json")
    session = InductionToolSession([])

    def raise_tool() -> dict:
        raise RuntimeError("fixture failure")

    session.list_transitions = raise_tool
    result = dispatch_tool(
        session,
        "list_transitions",
        "{}",
        receipt_transport=transport,
        decision_point_identity="decision-error",
    )
    assert result["ok"] is False
    receipt_identity = transport.last_receipt_identity
    row = transport.row(receipt_identity)
    assert row["gap_kind"] == "tool_error"
    assert row["response_transported"] is True
    assert row["agent_visible"] is False
    assert row["join_complete"] is False

    assert transport.record_delivery(receipt_identity, visible_text=json.dumps(result)) is True
    delivered = transport.row(receipt_identity)
    assert delivered["agent_visible"] is True
    assert delivered["response_used"] is None
    assert "agent_delivery" in delivered["hops"]


def test_scenario_arc_6859_unused_response_next_action_and_exact_outcome(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-6859-EXACT-OUTCOME-AND-CAUSAL-ELIGIBILITY keeps use distinct."""

    transport = _transport(tmp_path / "receipts.json")
    receipt_identity = _unknown_gap(transport)
    transport.record_delivery(receipt_identity, visible_text="unknown tool response")
    transport.record_next_action(
        action=(2, {"x": 4, "y": 5}),
        level_before=0,
        response_used=False,
        action_changed=False,
    )
    transport.record_exact_outcome(level_after=1, state_sha256="sha256:" + "a" * 64)

    row = transport.row(receipt_identity)
    assert row["response_used"] is False
    assert row["action_changed"] is False
    assert row["progress"] == 1
    assert row["exact_outcome"] is True
    assert row["join_complete"] is True
    assert row["causal_eligible"] is False
    assert set(row["hops"]) == set(receipt.REQUIRED_HOPS)


@dataclass
class Frame:
    levels_completed: int


def test_scenario_arc_6859_real_next_move_seam_records_action_and_later_outcome(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-6859-VISIBILITY-USE-AND-NEXT-ACTION uses E3AgentPolicy.next_move."""

    transport = _transport(tmp_path / "receipts.json")
    receipt_identity = _unknown_gap(transport)
    transport.record_delivery(receipt_identity, visible_text="visible")
    action = (3, None)
    policy = object.__new__(E3AgentPolicy)
    policy._first_party_tool_gap_receipt_transport = transport
    policy._provenance = None
    policy._next_move_routed = lambda frames, latest: action
    policy._maybe_apply_trace_automaton_action = lambda move, latest: move
    policy.record_target_licensed_route_shadow = lambda move, **kwargs: move
    policy.record_typed_obligation_shadow_monitor = lambda move, **kwargs: move
    policy._record_outcome_transport_proposal = lambda *args: None

    assert policy.next_move([], Frame(0)) is action
    assert "next_action" in transport.row(receipt_identity)["hops"]
    assert policy.next_move([], Frame(1)) is action
    row = transport.row(receipt_identity)
    assert row["progress"] == 1
    assert row["join_complete"] is True


def test_scenario_arc_6859_restart_and_duplicate_handling(tmp_path: Path) -> None:
    """SCENARIO-ARC-6859-PERSISTENCE-RESTART-AND-DEDUPLICATION is fail closed."""

    path = tmp_path / "receipts.json"
    first = _transport(path)
    receipt_identity = _unknown_gap(first)
    first.record_delivery(receipt_identity, visible_text="same")
    original = first.snapshot()

    restarted = _transport(path)
    assert restarted.snapshot()["rows"] == original["rows"]
    assert restarted.restart_loaded_count == 1
    assert restarted.record_delivery(receipt_identity, visible_text="same") is False
    assert restarted.snapshot()["deduplication"]["byte_identical_count"] == 1
    assert restarted.record_delivery(receipt_identity, visible_text="different") is False
    row = restarted.row(receipt_identity)
    assert row["quarantine_reason"] == "conflicting_duplicate_hop"
    assert row["join_complete"] is False
    assert restarted.snapshot()["deduplication"]["conflicting_count"] == 1


def test_req_arc_6859_factory_enables_persistent_transport(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-6859: opt-in construction uses the configured persistent path."""

    path = tmp_path / "enabled.json"
    monkeypatch.setenv(receipt.ENABLE_ENV, "1")
    monkeypatch.setenv(receipt.PATH_ENV, str(path))
    transport = receipt.maybe_make_first_party_tool_gap_receipt_transport("g", "run")
    assert transport is not None
    assert path.exists()
    assert transport.snapshot()["schema"] == receipt.RECEIPT_SCHEMA


def test_req_arc_6859_hash_and_payload_defensive_paths(tmp_path: Path) -> None:
    """REQ-ARC-6859: hashes cover action objects, fallback objects, and missing files."""

    @dataclass
    class Action:
        action_id: int
        data: dict[str, int]

    class Value:
        def __str__(self) -> str:
            return "stable-value"

    assert receipt._now().endswith("Z")
    assert receipt.canonical_sha256(Action(6, {"x": 1})).startswith("sha256:")
    assert receipt.canonical_sha256(Value()).startswith("sha256:")
    assert receipt.sha256_file(tmp_path / "missing").startswith("sha256:")
    with pytest.raises(ValueError, match="unknown provenance"):
        receipt.FirstPartyToolGapReceiptTransport(
            tmp_path / "bad.json",
            attempt_identity="a",
            game_id="g",
            provenance_class="unknown",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "changed", "schema changed"),
        ("attempt_identity", "changed", "attempt identity changed"),
        ("game_id", "changed", "game identity changed"),
        ("rows", {}, "rows must be a list"),
    ],
)
def test_req_arc_6859_restart_rejects_changed_identity_or_schema(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """REQ-ARC-6859: restart fails closed on incompatible persisted bytes."""

    path = tmp_path / f"{field}.json"
    transport = _transport(path)
    document = transport.snapshot()
    document[field] = value
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _transport(path)


def test_req_arc_6859_bad_arguments_and_default_output_path(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-6859: bad signatures are gaps and opt-in has a safe default path."""

    transport = _transport(tmp_path / "bad-arguments.json")
    result = dispatch_tool(
        InductionToolSession([]),
        "query_region",
        "{}",
        receipt_transport=transport,
        decision_point_identity="bad-arguments",
    )
    assert result["ok"] is False
    assert transport.row(transport.last_receipt_identity)["gap_kind"] == "bad_arguments"

    monkeypatch.setenv(receipt.ENABLE_ENV, "1")
    monkeypatch.delenv(receipt.PATH_ENV, raising=False)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    made = receipt.maybe_make_first_party_tool_gap_receipt_transport("g", "default-path")
    assert made is not None
    assert made.path.parent == tmp_path / "results" / "arc_tool_gap_receipts"
