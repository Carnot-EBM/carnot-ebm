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


class TestBudgetTimeoutSlack:
    """REQ-CONDUCTOR-RECEIPT-2: a caller timeout must outlast its own budget.

    SCENARIO-CONDUCTOR-RECEIPT-5 / -6. The budget is checked BETWEEN units, so a
    unit that starts just before the deadline still runs to completion. If the
    caller's kill timeout does not leave room for that unit plus the report
    write, the PARTIAL-report mechanism cannot fire and the audit dies silently
    -- which is what happened to the QA-layer audit at 750s budget / 900s
    timeout against units measured at 250-375s each.

    This reads the REAL conductor source rather than a fixture: the defect was
    two constants drifting apart, and only the real call sites can show that.
    """

    @staticmethod
    def _budgeted_call_sites() -> list[tuple[str, int, int]]:
        """(audit name, --budget-seconds, timeout=) for every budgeted call."""
        import ast

        tree = ast.parse((REPO_ROOT / "scripts" / "research_conductor.py").read_text())
        found: list[tuple[str, int, int]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", None) != "_run_audit_with_receipt":
                continue
            name = None
            if node.args and isinstance(node.args[0], ast.Constant):
                name = node.args[0].value
            timeout = None
            for kw in node.keywords:
                if kw.arg == "timeout" and isinstance(kw.value, ast.Constant):
                    timeout = kw.value.value
            budget = None
            for arg in node.args:
                if not isinstance(arg, ast.List):
                    continue
                parts = [e.value for e in arg.elts if isinstance(e, ast.Constant)]
                for i, part in enumerate(parts):
                    if part == "--budget-seconds" and i + 1 < len(parts):
                        budget = int(parts[i + 1])
            if budget is not None and timeout is not None:
                found.append((str(name), budget, timeout))
        return found

    def test_slack_constant_covers_a_worst_case_unit(self) -> None:
        # 375s was the worst measured QA-layer unit (2 units / 750s budget,
        # 2026-08-22). The constant must cover that plus the report write.
        assert rc.AUDIT_TIMEOUT_SLACK_S >= 375

    def test_every_budgeted_audit_has_slack(self) -> None:
        sites = self._budgeted_call_sites()
        # Guard the guard: if the AST walk finds nothing, this test would pass
        # vacuously and the invariant would go unchecked.
        assert sites, "no budgeted _run_audit_with_receipt call sites found"
        violations = [
            f"{name}: timeout={timeout} < budget={budget} + slack={rc.AUDIT_TIMEOUT_SLACK_S}"
            for name, budget, timeout in sites
            if timeout < budget + rc.AUDIT_TIMEOUT_SLACK_S
        ]
        assert not violations, "; ".join(violations)

    def test_qa_layer_budget_reviews_more_than_a_handful(self) -> None:
        # The origin incident was not only the timeout: a 750s budget against
        # 250-375s units could review 2-3 of 20 selected units, so rotation
        # crawled (offset 5 of 174). Assert the budget buys a meaningful slice.
        sites = {name: (budget, timeout) for name, budget, timeout in self._budgeted_call_sites()}
        budget, _ = sites["qa-layer-authenticity-audit"]
        assert budget // 375 >= 4, f"budget {budget}s reviews < 4 worst-case units"
