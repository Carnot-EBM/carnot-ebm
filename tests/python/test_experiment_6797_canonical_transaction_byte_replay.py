"""Tests for the snapshot-complete Exp6791 deterministic replay.

Spec refs: REQ-CL-6797 and SCENARIO-CL-6797-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6791_compositional_online_constraint_routing_ab as source_code
from carnot import experiment_6797_canonical_transaction_byte_replay as exp
from carnot.durable_row_checkpoint import DurableRowCheckpoint, complete_row_envelope


def _commit_pair(tmp_path: Path) -> tuple[list[dict], Path]:
    state_root = tmp_path / "stores"
    store = source_code.IsolatedTransactionStore(
        state_root / "order_1" / "compositional_online",
        "compositional_online",
        "order_1",
    )
    receipts = []
    predecessor = exp.genesis_predecessor(
        order_id="order_1",
        arm="compositional_online",
        parent_hash=store.state_hash(),
    )
    for index in (1, 2):
        raw = store.commit_factor(
            {"factor_id": f"factor-{index}", "value": index},
            transaction_id=f"tx:order_1:event-{index}:compositional_online",
        )
        receipt = exp.capture_commit_receipt(
            raw,
            order_id="order_1",
            arm="compositional_online",
            event_id=f"event-{index}",
            position=index,
            chain_index=index,
            chain_predecessor=predecessor,
        )
        receipts.append(receipt)
        predecessor = receipt["receipt_hash"]
    return receipts, state_root


def test_scenario_cl_6797_canonical_bytes_round_trip_and_hash(tmp_path: Path) -> None:
    """SCENARIO-CL-6797-CANONICAL-BYTES verifies both exact snapshots."""

    first, _ = _commit_pair(tmp_path)
    receipt = first[0]
    parent = exp.decode_snapshot(receipt["parent_state_bytes"])
    new = exp.decode_snapshot(receipt["new_state_bytes"])

    assert exp.encode_snapshot(parent) == receipt["parent_state_bytes"]
    assert exp.encode_snapshot(new) == receipt["new_state_bytes"]
    assert exp.sha256_bytes(parent) == receipt["parent_hash"]
    assert exp.sha256_bytes(new) == receipt["new_state_hash"]
    assert receipt["parent_hash_verified_at_commit"] is True
    assert receipt["new_state_hash_verified_at_commit"] is True
    assert exp.verify_commit_receipts(first) == []


def test_scenario_cl_6797_byte_chain_rejects_corruption(tmp_path: Path) -> None:
    """SCENARIO-CL-6797-BYTE-CHAIN refuses changed bytes and ordering."""

    receipts, _ = _commit_pair(tmp_path)
    flipped = deepcopy(receipts)
    changed_bytes = exp.decode_snapshot(flipped[0]["new_state_bytes"]).replace(
        b'"version":1', b'"version":9'
    )
    flipped[0]["new_state_bytes"] = exp.encode_snapshot(changed_bytes)
    reordered = list(reversed(deepcopy(receipts)))
    stale = deepcopy(receipts)
    stale[1]["parent_state_bytes"] = stale[0]["parent_state_bytes"]
    stale[1]["parent_hash"] = stale[0]["parent_hash"]

    assert "new_state_hash_mismatch" in exp.verify_commit_receipts(flipped)
    assert "chain_position_not_increasing" in exp.verify_commit_receipts(reordered)
    assert "parent_bytes_do_not_extend_chain" in exp.verify_commit_receipts(stale)


def test_scenario_cl_6797_attacks_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-6797-ATTACKS preserves accepted bytes for eight attacks."""

    receipts, state_root = _commit_pair(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = DurableRowCheckpoint(checkpoint_path, {"manifest": "frozen"})
    checkpoint.append(
        complete_row_envelope(
            row_id="order_1",
            manifest_hash=checkpoint.manifest_hash,
            payload={"cell_count": 2},
            attempt=1,
            start_receipt={"phase": "start"},
            end_receipt={"phase": "complete"},
        )
    )
    before = exp.sha256_bytes(exp.canonical_json_bytes(receipts))
    attacks = exp.run_attack_suite(
        receipts,
        checkpoint_path=checkpoint_path,
        checkpoint_manifest={"manifest": "frozen"},
        state_root=state_root,
    )

    assert [row["attack_id"] for row in attacks] == list(exp.ATTACK_IDS)
    assert all(row["failed_closed"] is True for row in attacks)
    assert all(row["committed_bytes_unchanged"] is True for row in attacks)
    assert exp.sha256_bytes(exp.canonical_json_bytes(receipts)) == before


def test_scenario_cl_6797_identity_reduces_source_rows() -> None:
    """SCENARIO-CL-6797-IDENTITY recomputes frozen activity from rows."""

    source = json.loads(exp.EXP6791_PATH.read_text(encoding="utf-8"))
    checks = exp.replay_identity_checks(
        source,
        deepcopy(source["rows"]),
        deepcopy(source["transaction_receipts"]),
    )

    assert checks["all_passed"] is True
    assert checks["observed_counts"] == {
        "compositional_online_writes": 1063,
        "compositional_online_later_reads": 3132,
        "compositional_online_action_changes": 721,
        "committed_transactions": 3189,
    }
    changed = deepcopy(source["rows"])
    changed[0]["selected_action"] = "not-a-route"
    assert (
        exp.replay_identity_checks(source, changed, source["transaction_receipts"])["all_passed"]
        is False
    )


def test_scenario_cl_6797_fresh_resume_skips_prefix_once(tmp_path: Path) -> None:
    """SCENARIO-CL-6797-FRESH-RESUME resumes four pending frozen orders."""

    manifest = {"order_ids": ["order_1", "order_2"]}
    launches: list[tuple[list[str], int | None]] = []

    def fake_launcher(
        order_ids: list[str], attempt: int, stop_after: int | None
    ) -> tuple[list[dict], dict]:
        launches.append((order_ids, stop_after))
        emitted = order_ids[:stop_after] if stop_after is not None else order_ids
        payloads = [
            {
                "order_id": order_id,
                "rows": [{"row_key": f"{order_id}:cell", "order_id": order_id}],
                "transaction_receipts": [],
            }
            for order_id in emitted
        ]
        return payloads, {
            "attempt": attempt,
            "pid": 100 + attempt,
            "fresh_process": True,
            "interrupted": stop_after is not None,
        }

    rows, transactions, receipts = exp.execute_checkpointed_orders(
        checkpoint_path=tmp_path / "orders.json",
        checkpoint_manifest=manifest,
        ordered_ids=manifest["order_ids"],
        launcher=fake_launcher,
        cells_per_order=1,
    )

    assert launches == [(["order_1", "order_2"], 1), (["order_2"], None)]
    assert [row["order_id"] for row in rows] == ["order_1", "order_2"]
    assert transactions == []
    assert receipts["interrupted_prefix_cell_count"] == 1
    assert receipts["skipped_complete_cells_exactly_once"] is True
    assert receipts["fresh_process_resume"] is True


def test_scenario_cl_6797_blocked_precondition_has_no_rows(tmp_path: Path) -> None:
    """SCENARIO-CL-6797-BLOCKED stops before workers or reduced replay."""

    artifact = exp.run_experiment(
        run_date="20260831",
        checkpoint_path=tmp_path / "checkpoint.json",
        state_root=tmp_path / "stores",
        precondition_overrides={"sufficient_disk_bytes": False},
        duration_s=0.25,
    )

    assert artifact["status"] == "complete_blocked_transaction_byte_replay"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["transaction_receipts"] == []
    assert artifact["gate_check_summary"]["failed_checks"] == ["sufficient_disk_bytes"]
    assert exp.validate_artifact(artifact) == []


def test_req_cl_6797_checkpoint_refuses_changed_manifest(tmp_path: Path) -> None:
    """REQ-CL-6797 binds durable resume to the complete frozen manifest."""

    checkpoint = DurableRowCheckpoint(tmp_path / "checkpoint.json", {"version": 1})
    good_bytes = checkpoint.path.read_bytes()

    with pytest.raises(exp.ManifestMismatchError):
        DurableRowCheckpoint(checkpoint.path, {"version": 2})

    assert checkpoint.path.read_bytes() == good_bytes


def test_req_cl_6797_validator_rejects_count_and_checksum_drift(tmp_path: Path) -> None:
    """REQ-CL-6797 derives readiness from byte and identity evidence."""

    artifact = exp.run_experiment(
        run_date="20260831",
        checkpoint_path=tmp_path / "checkpoint.json",
        state_root=tmp_path / "stores",
        precondition_overrides={"canonical_serializer": False},
        duration_s=0.5,
    )
    bad_count = deepcopy(artifact)
    bad_count["committed_transaction_count"] = 1
    bad_count["reproducibility_checksum"] = exp.reproducibility_checksum(bad_count)
    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"

    assert "blocked artifact has nonzero snapshot counts" in exp.validate_artifact(bad_count)
    assert "reproducibility checksum mismatch" in exp.validate_artifact(bad_checksum)
