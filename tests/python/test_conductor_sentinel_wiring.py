"""Wiring tests: the conductor actually calls the self-supervision tools.

REQ-CONDUCTOR-SENTINEL-3 / REQ-OPS-AUDIT-LEDGER-1. A check nothing calls is
the bug class (the QA-layer audit's --check-targets lesson): the sentinel
and the ledger must be invoked from the conductor through the receipt
checker, with the receipt being each tool's own always-rewritten state file
— never an exit code, and never a file that only changes on findings.

These are source-level wiring assertions plus receipt-contract tests. They
deliberately do NOT call research_step() — it touches git and the live
roadmap; the tools' own behavior is covered by their dedicated test files.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import research_conductor as rc  # noqa: E402


def test_receipt_constants_point_at_state_files():
    assert rc.RUN_SENTINEL_STATE == rc.PROJECT_ROOT / "ops" / ".run_sentinel_state.json"
    assert rc.AUDIT_LEDGER_STATE == rc.PROJECT_ROOT / "ops" / ".audit_findings_ledger_state.json"


def _code_only(source: str) -> str:
    """Source minus comment lines, so a commented-out call reads as absent
    (adversarial-review finding 5, 2026-08-22: substring asserts against
    raw getsource stayed green when the call was merely commented out)."""
    return "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))


def test_research_step_invokes_sentinel_with_receipt():
    """The sentinel call sits inside research_step, receipt-checked against
    RUN_SENTINEL_STATE (the file the sentinel rewrites on every scan)."""
    source = _code_only(inspect.getsource(rc.research_step))
    assert '"run-sentinel"' in source
    assert "conductor_run_sentinel.py" in source
    assert "RUN_SENTINEL_STATE" in source


def test_milestone_close_invokes_ledger_with_receipt():
    source = _code_only(inspect.getsource(rc.research_step))
    assert '"audit-findings-ledger"' in source
    assert "audit_findings_ledger.py" in source
    assert "AUDIT_LEDGER_STATE" in source


def test_sentinel_state_file_is_rewritten_on_a_clean_scan(tmp_path):
    """The receipt contract end-to-end: a scan with zero findings still
    rewrites the state file, so _run_audit_with_receipt sees a fresh mtime.
    A tool whose receipt only moves on findings would read as dead on every
    healthy day — the inverse failure."""
    spec = importlib.util.spec_from_file_location(
        "conductor_run_sentinel", REPO_ROOT / "scripts" / "conductor_run_sentinel.py"
    )
    sentinel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sentinel)
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    state = tmp_path / "state.json"
    sentinel.run_scan(
        proc_root=proc_root,
        gpu_runner=lambda args: "0, GPU-aaa, 4, 24576\n" if "query-gpu" in args[0] else "",
        pin_loader=lambda: ("Qwen3.8-27B", None),
        conductor_log=tmp_path / "log.md",
        known_issues=tmp_path / "ki.md",
        state_path=state,
        default_log_dir=tmp_path / "no_logs",
    )
    first = json.loads(state.read_text())["last_scan_utc"]
    assert first
    # And the ledger's receipt behaves the same on a no-op run.
    spec2 = importlib.util.spec_from_file_location(
        "audit_findings_ledger", REPO_ROOT / "scripts" / "audit_findings_ledger.py"
    )
    ledger = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(ledger)
    lstate = tmp_path / "lstate.json"
    ledger.run(
        # The complete-map form. `report_path=` alone would isolate the claim
        # audit and silently read the REAL tracked QA-layer report.
        report_paths={n: tmp_path / "absent.md" for n in (ledger.AUDIT_NAME, ledger.QA_AUDIT_NAME)},
        ledger_path=tmp_path / "ledger.md",
        conductor_log=tmp_path / "log.md",
        known_issues=tmp_path / "ki.md",
        state_path=lstate,
    )
    assert json.loads(lstate.read_text())["last_scan_utc"]
