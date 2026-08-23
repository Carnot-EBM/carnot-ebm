"""Tests for the reusable atomic shard transaction.

Spec refs: REQ-BENCH-6514, SCENARIO-BENCH-6514-SHARD-IDENTITY,
SCENARIO-BENCH-6514-PLANNED-TERMINAL, SCENARIO-BENCH-6514-RESUME-CRASHES,
SCENARIO-BENCH-6514-CORRUPT-QUARANTINE, SCENARIO-BENCH-6514-ATOMIC-REPLACE,
SCENARIO-BENCH-6514-CONCURRENCY, SCENARIO-BENCH-6514-CLOSED-FAILURE.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from carnot.atomic_shard_transaction import (
    AtomicShardTransaction,
    ConcurrentWriterError,
    CorruptShardError,
    CrashInjected,
    CrashPlan,
    DuplicateUnitError,
    InsufficientDiskError,
    MissingTerminalUnitError,
    canonical_json_bytes,
    nonterminal_status_reason,
    sha256_bytes,
    sha256_json,
)


def _payload(name: str = "ok") -> dict[str, Any]:
    return {
        "status": "complete_ready",
        "honest_verdict": f"complete_{name}",
        "verdict_class": "null",
        "rows": [{"unit_id": "u1", "value": name}],
    }


def _transaction(tmp_path: Path, **kwargs: Any) -> AtomicShardTransaction:
    return AtomicShardTransaction(
        work_dir=tmp_path / "work",
        final_path=tmp_path / "final.json",
        transaction_id="test-tx",
        **kwargs,
    )


def test_scenario_bench_6514_shard_identity_and_duplicate_refusal(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6514-SHARD-IDENTITY: content hashes own unit rows."""

    with _transaction(tmp_path) as tx:
        tx.plan_units(["u1"])
        assert tx.plan_units(["u1"]) == []
        first = tx.write_terminal_unit("u1", {"value": 1}, disposition="success")
        second = tx.write_terminal_unit("u1", {"value": 1}, disposition="success")
        tx.journal_path.write_text(
            tx.journal_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )

        assert first["shard_hash"] == sha256_json({"value": 1})
        assert first["shard_path"].endswith(f"{first['shard_hash'][7:]}.json")
        assert first["idempotent"] is False
        assert second["idempotent"] is True
        assert second["shard_hash"] == first["shard_hash"]
        assert all(row["record_hash"].startswith("sha256:") for row in tx.read_journal())

        with pytest.raises(DuplicateUnitError, match="different content"):
            tx.write_terminal_unit("u1", {"value": 2}, disposition="success")

        data = canonical_json_bytes({"value": 3})
        third_hash = sha256_bytes(data)
        tx._write_file_atomically(tx._shard_path(third_hash), data)
        tx._append_journal_record(
            {
                "schema": "carnot.atomic_shard_transaction.v1",
                "transaction_id": "test-tx",
                "record_type": "terminal_unit",
                "unit_id": "u1",
                "disposition": "success",
                "shard_hash": third_hash,
                "shard_path": str(tx._shard_path(third_hash)),
                "recorded_unix_s": 1.0,
            }
        )
        with pytest.raises(DuplicateUnitError, match="conflicting journal hashes"):
            tx.resume_state()


def test_scenario_bench_6514_planned_units_must_close_before_replace(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6514-PLANNED-TERMINAL: missing units block finalization."""

    with _transaction(tmp_path) as tx:
        tx.plan_units(["u1", "u2"])
        tx.write_terminal_unit("u1", {"value": 1})

        with pytest.raises(MissingTerminalUnitError, match="u2"):
            tx.finalize(_payload())
        assert not (tmp_path / "final.json").exists()

        tx.write_terminal_unit("u2", {"value": 2})
        bad_payload = {"status": "running_bootstrap", "honest_verdict": "running"}
        assert nonterminal_status_reason(bad_payload) == "status=running_bootstrap"
        assert (
            nonterminal_status_reason({"status": "mystery", "honest_verdict": "complete_x"})
            == "status=mystery"
        )
        assert (
            nonterminal_status_reason({"status": "complete_x", "honest_verdict": "mystery"})
            == "honest_verdict=mystery"
        )
        with pytest.raises(ValueError, match="nonterminal final payload"):
            tx.finalize(bad_payload)
        assert not (tmp_path / "final.json").exists()

        receipt = tx.finalize(_payload("closed"))
        loaded = json.loads((tmp_path / "final.json").read_text(encoding="utf-8"))

        assert loaded["status"] == "complete_ready"
        assert receipt["atomic_replace"] is True
        assert receipt["file_fsync"] is True
        assert receipt["directory_fsync_attempted"] is True
        assert tx.resume_state()["missing_unit_ids"] == []


@pytest.mark.parametrize(
    "stage",
    [
        "before_shard_write",
        "after_shard_write",
        "during_journal_update",
        "before_replace",
        "after_replace",
    ],
)
def test_scenario_bench_6514_crash_points_resume_safely(
    tmp_path: Path,
    stage: str,
) -> None:
    """SCENARIO-BENCH-6514-RESUME-CRASHES: each crash point has a safe resume."""

    crash = CrashPlan.once(stage)
    with _transaction(tmp_path, crash_plan=crash) as tx:
        tx.plan_units(["u1"])
        if stage in {"before_shard_write", "after_shard_write", "during_journal_update"}:
            with pytest.raises(CrashInjected, match=stage):
                tx.write_terminal_unit("u1", {"value": stage})
        else:
            tx.write_terminal_unit("u1", {"value": stage})
            with pytest.raises(CrashInjected, match=stage):
                tx.finalize(_payload(stage))

    with _transaction(tmp_path) as resumed:
        state = resumed.resume_state()
        if stage == "after_replace":
            loaded = json.loads((tmp_path / "final.json").read_text(encoding="utf-8"))
            assert loaded["status"] == "complete_ready"
            assert state["missing_unit_ids"] == []
        else:
            if stage in {"after_shard_write", "during_journal_update"}:
                assert state["orphan_shard_hashes"]
            resumed.write_terminal_unit("u1", {"value": stage})
            receipt = resumed.finalize(_payload(f"{stage}_resumed"))
            assert receipt["atomic_replace"] is True
            loaded = json.loads((tmp_path / "final.json").read_text(encoding="utf-8"))
            assert loaded["honest_verdict"].startswith("complete_")


def test_scenario_bench_6514_corrupt_shard_quarantines_and_rewrites(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6514-CORRUPT-QUARANTINE: bad bytes cannot resume."""

    with _transaction(tmp_path) as tx:
        tx.plan_units(["u1"])
        receipt = tx.write_terminal_unit("u1", {"value": "stable"})
        shard_path = Path(receipt["shard_path"])

    shard_path.write_text('{"value":"corrupt"}\n', encoding="utf-8")

    with _transaction(tmp_path) as resumed:
        state = resumed.resume_state()
        assert state["corrupt_shard_rows"][0]["expected_hash"] == receipt["shard_hash"]
        assert state["missing_unit_ids"] == ["u1"]
        assert not shard_path.exists()
        assert Path(state["corrupt_shard_rows"][0]["quarantine_path"]).exists()

        rewritten = resumed.write_terminal_unit("u1", {"value": "stable"})
        assert rewritten["shard_hash"] == receipt["shard_hash"]
        resumed.finalize(_payload("after_corruption"))
        assert json.loads((tmp_path / "final.json").read_text(encoding="utf-8"))["status"]


def test_scenario_bench_6514_corrupt_journal_record_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6514-CORRUPT-QUARANTINE: journal hashes are verified."""

    with _transaction(tmp_path) as tx:
        tx.plan_units(["u1"])
        tx.write_terminal_unit("u1", {"value": "stable"})

    journal = tmp_path / "work" / "journal.jsonl"
    lines = journal.read_text(encoding="utf-8").splitlines()
    broken = json.loads(lines[-1])
    broken["unit_id"] = "evil"
    lines[-1] = json.dumps(broken, sort_keys=True)
    journal.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with _transaction(tmp_path) as resumed:
        with pytest.raises(CorruptShardError, match="journal record hash mismatch"):
            resumed.resume_state()


def test_scenario_bench_6514_concurrency_stale_lock_disk_and_failure_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6514-CONCURRENCY/CLOSED-FAILURE: unsafe paths close."""

    tx = _transaction(tmp_path)
    tx.begin()
    try:
        with pytest.raises(ConcurrentWriterError, match="active lock"):
            _transaction(tmp_path).begin()
    finally:
        tx.close()

    lock_path = tmp_path / "work" / "LOCK"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": 999999999, "transaction_id": "test-tx"}) + "\n",
        encoding="utf-8",
    )
    old = 1
    os.utime(lock_path, (old, old))
    with _transaction(tmp_path, stale_lock_s=0.01) as recovered:
        assert recovered.lock_receipt["stale_lock_recovered"] is True

    with _transaction(tmp_path, min_free_bytes=10**30) as disk_tx:
        disk_tx.plan_units(["u1"])
        with pytest.raises(InsufficientDiskError, match="insufficient disk"):
            disk_tx.write_terminal_unit("u1", {"large": "payload"})

    with _transaction(tmp_path) as replace_tx:
        replace_tx.plan_units(["u1"])
        with patch("os.replace", side_effect=OSError("replace failed")):
            with pytest.raises(OSError, match="replace failed"):
                replace_tx.write_terminal_unit("u1", {"value": "rename_failure"})

    failure_payload = {
        "status": "blocked_atomic_shard_transaction",
        "honest_verdict": "blocked_atomic_shard_transaction: forced diagnostic",
        "verdict_class": "blocked",
        "diagnostics": [{"check": "forced", "observed": "bad"}],
    }
    with _transaction(tmp_path) as failure_tx:
        receipt = failure_tx.write_failure_artifact(failure_payload)
        assert receipt["failure_artifact_written"] is True
        assert json.loads((tmp_path / "final.json").read_text(encoding="utf-8"))[
            "status"
        ].startswith("blocked_")

        preserved = failure_tx.write_failure_artifact(
            {
                "status": "blocked_atomic_shard_transaction",
                "honest_verdict": "blocked_atomic_shard_transaction: second",
                "verdict_class": "blocked",
            },
            preserve_existing_terminal=True,
        )
    assert preserved["existing_terminal_preserved"] is True

    with _transaction(tmp_path) as failure_tx:
        with pytest.raises(ValueError, match="nonterminal failure payload"):
            failure_tx.write_failure_artifact(
                {"status": "running_bootstrap", "honest_verdict": "running"},
                preserve_existing_terminal=False,
            )
