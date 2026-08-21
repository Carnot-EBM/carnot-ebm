"""Tests for audit run receipts: caller-side freshness check + QA-audit budget.

REQ: REQ-CONDUCTOR-RECEIPT-1 (openspec/capabilities/research-harnesses/spec.md).
SCENARIOs: SCENARIO-CONDUCTOR-RECEIPT-1 (stale receipt writes a BLOCK line),
SCENARIO-CONDUCTOR-RECEIPT-2 (fresh receipt passes), SCENARIO-CONDUCTOR-RECEIPT-3
(budget produces a PARTIAL report), SCENARIO-CONDUCTOR-RECEIPT-4 (rotation
advances only with the receipt, by the completed count).

Origin: the QA-layer audit wrote rotation state before its LLM loop and the
report only after ALL units; the conductor killed it at timeout=900 with
check=False. No report landed after 2026-07-29 while the rotation offset
advanced 20 units per close — coverage accounting without coverage, and
nothing warned durably.

All file paths are redirected to tmp_path — no test writes tracked state.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pytest  # noqa: E402

import research_conductor as rc  # noqa: E402
import scripts.qa_layer_authenticity_audit as qla  # noqa: E402


class TestRunAuditWithReceipt:
    """Conductor-side receipt verification (_run_audit_with_receipt)."""

    def _wire(self, tmp_path: Path, monkeypatch) -> Path:
        (tmp_path / "ops").mkdir()
        log = tmp_path / "ops" / "conductor-log.md"
        monkeypatch.setattr(rc, "CONDUCTOR_LOG", log)
        monkeypatch.setattr(rc, "PROJECT_ROOT", tmp_path)
        return log

    def test_fresh_receipt_passes(self, tmp_path: Path, monkeypatch) -> None:
        # SCENARIO-CONDUCTOR-RECEIPT-2: the subprocess rewrites its report
        # during the call — the caller records success, no BLOCK line.
        log = self._wire(tmp_path, monkeypatch)
        receipt = tmp_path / "ops" / "report.md"
        writer = f"import pathlib; pathlib.Path({str(receipt)!r}).write_text('r')"
        ok = rc._run_audit_with_receipt(
            "fake-audit", [sys.executable, "-c", writer], receipt=receipt, timeout=30
        )
        assert ok is True
        assert not log.exists() or "BLOCK" not in log.read_text()

    def test_stale_receipt_blocks(self, tmp_path: Path, monkeypatch) -> None:
        # SCENARIO-CONDUCTOR-RECEIPT-1: exit 0 but the report was NOT
        # rewritten — exactly the 23-day silent-death shape. The exit code
        # must not be believed; the stale receipt writes a durable BLOCK.
        log = self._wire(tmp_path, monkeypatch)
        receipt = tmp_path / "ops" / "report.md"
        receipt.write_text("old report")
        old = time.time() - 3600
        os.utime(receipt, (old, old))
        ok = rc._run_audit_with_receipt(
            "fake-audit", [sys.executable, "-c", "pass"], receipt=receipt, timeout=30
        )
        assert ok is False
        text = log.read_text()
        assert "Audit receipt STALE: fake-audit" in text
        assert "BLOCK" in text

    def test_missing_receipt_blocks(self, tmp_path: Path, monkeypatch) -> None:
        log = self._wire(tmp_path, monkeypatch)
        ok = rc._run_audit_with_receipt(
            "fake-audit",
            [sys.executable, "-c", "pass"],
            receipt=tmp_path / "ops" / "never_written.md",
            timeout=30,
        )
        assert ok is False
        assert "BLOCK" in log.read_text()

    def test_timeout_blocks(self, tmp_path: Path, monkeypatch) -> None:
        log = self._wire(tmp_path, monkeypatch)
        ok = rc._run_audit_with_receipt(
            "fake-audit",
            [sys.executable, "-c", "import time; time.sleep(30)"],
            receipt=tmp_path / "ops" / "report.md",
            timeout=1,
        )
        assert ok is False
        assert "timeout" in log.read_text()

    def test_no_receipt_path_uses_exit_code(self, tmp_path: Path, monkeypatch) -> None:
        log = self._wire(tmp_path, monkeypatch)
        ok = rc._run_audit_with_receipt(
            "fake-sweep", [sys.executable, "-c", "pass"], receipt=None, timeout=30
        )
        assert ok is True
        bad = rc._run_audit_with_receipt(
            "fake-sweep", [sys.executable, "-c", "raise SystemExit(3)"], receipt=None, timeout=30
        )
        assert bad is False
        assert "rc=3" in log.read_text()


class TestQaAuditBudgetAndRotation:
    """QA-layer audit: --budget-seconds PARTIAL report + rotation-after-receipt."""

    def _wire(self, tmp_path: Path, monkeypatch, argv: list[str]) -> dict:
        (tmp_path / "ops").mkdir()
        monkeypatch.setattr(qla, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(qla, "REPORT_PATH", tmp_path / "ops" / "qa_report.md")
        monkeypatch.setattr(qla, "all_target_paths", lambda: [tmp_path / "fake_target.py"])
        monkeypatch.setattr(
            qla, "build_units", lambda p: [(f"u{i}", "body", "prompt") for i in range(7)]
        )
        monkeypatch.setattr(qla, "discover_unaudited_guards", lambda *a, **k: [])
        clock = {"t": 0.0, "ran": []}

        def fake_run_one(label, body, prompt, args, out, counts, flagged, voids, missed=None):
            clock["ran"].append(label)
            clock["t"] += 6.0  # each unit "takes" 6 fake seconds
            counts["CLEAN"] = counts.get("CLEAN", 0) + 1
            out.append(f"## {label}\n")

        monkeypatch.setattr(qla, "_run_one", fake_run_one)
        monkeypatch.setattr(qla, "_now", lambda: clock["t"])
        monkeypatch.setattr(sys, "argv", ["qa_layer_authenticity_audit.py", *argv])
        return clock

    def test_budget_truncates_with_partial_report(self, tmp_path: Path, monkeypatch) -> None:
        # SCENARIO-CONDUCTOR-RECEIPT-3: budget 10s, 6 fake seconds per unit
        # -> units 1..2 run (t=0, t=6), unit 3 sees t=12 >= 10 and stops.
        clock = self._wire(
            tmp_path, monkeypatch, ["--limit", "5", "--budget-seconds", "10", "--model", "claude"]
        )
        assert qla.main() == 0
        assert clock["ran"] == ["u0", "u1"]
        report = (tmp_path / "ops" / "qa_report.md").read_text()
        assert "PARTIAL RUN" in report
        assert "2 of 5" in report
        # SCENARIO-CONDUCTOR-RECEIPT-4: rotation advanced by the COMPLETED
        # count (2), not the requested --limit (5).
        state = json.loads((tmp_path / "ops" / ".qa_layer_audit_rotation.json").read_text())
        assert state["offset"] == 2

    def test_full_run_advances_by_limit(self, tmp_path: Path, monkeypatch) -> None:
        clock = self._wire(tmp_path, monkeypatch, ["--limit", "5", "--model", "claude"])
        assert qla.main() == 0
        assert clock["ran"] == ["u0", "u1", "u2", "u3", "u4"]
        report = (tmp_path / "ops" / "qa_report.md").read_text()
        assert "PARTIAL RUN" not in report
        state = json.loads((tmp_path / "ops" / ".qa_layer_audit_rotation.json").read_text())
        assert state["offset"] == 5  # (0 + 5) % 7

    def test_rotation_not_advanced_when_report_write_fails(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # SCENARIO-CONDUCTOR-RECEIPT-4, the ordering half: no receipt, no
        # consumption. Making REPORT_PATH a directory forces the report
        # write to raise AFTER the units ran; the rotation offset must
        # stay unwritten so the next run re-covers the slice.
        self._wire(tmp_path, monkeypatch, ["--limit", "3", "--model", "claude"])
        (tmp_path / "ops" / "qa_report.md").mkdir()
        with pytest.raises(IsADirectoryError):
            qla.main()
        assert not (tmp_path / "ops" / ".qa_layer_audit_rotation.json").exists()
