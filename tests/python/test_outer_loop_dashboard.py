"""REQ-INFRA-6840: the hourly outer-loop dashboard reports measured state, never recalled state.

The hourly status report drifted from a compact scannable block into paragraphs, and the
operator asked for the block back. A script fixes that permanently -- same fields, same order,
every hour -- and removes two error classes this session hit repeatedly: elapsed time estimated
rather than computed (days-old events reported as months old), and process liveness assumed
rather than read.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "outer_loop_dashboard.py"


def _module():
    spec = importlib.util.spec_from_file_location("_dash", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_dash"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_age_is_computed_not_estimated() -> None:
    """The whole point: 8 days must read as 8, not as "a while back"."""
    m = _module()
    now = datetime(2026, 8, 30, tzinfo=UTC)
    assert m.days_since((now - timedelta(days=8)).isoformat(), now) == 8
    assert m.days_since((now - timedelta(days=1)).isoformat(), now) == 1


def test_an_unparseable_date_returns_none_rather_than_a_guess() -> None:
    """A wrong age is worse than an absent one -- the wrong one gets repeated as fact."""
    assert _module().days_since("not-a-date") is None


def test_liveness_is_read_from_proc_not_assumed() -> None:
    m = _module()
    assert m.pid_alive(os.getpid())
    assert not m.pid_alive(999_999_999)


def test_outcomes_are_counted_from_the_log(tmp_path, monkeypatch) -> None:
    """Counted, never recalled. A remembered outcome mix is how a bad day gets called good."""
    m = _module()
    monkeypatch.setattr(m, "REPO", tmp_path)
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "conductor-log.md").write_text(
        "| 2026-08-30 01:00 UTC | a | OK | x |\n"
        "| 2026-08-30 02:00 UTC | b | FAIL | y |\n"
        "| 2026-08-30 03:00 UTC | c | OK | z |\n"
        "| 2026-08-29 01:00 UTC | d | OK | other day |\n"
    )
    assert m.outcome_mix("2026-08-30") == {"OK": 2, "FAIL": 1}


def test_a_dead_job_with_no_receipt_says_so_explicitly(tmp_path) -> None:
    """The 2026-08-29 case: a 7-hour job vanished and nothing said what killed it.

    A missing receipt is itself evidence (SIGKILL, OOM, kernel) and must be reported as such
    rather than shown as a blank.
    """
    out = _module().render([("ab", 999_999_999, tmp_path / "absent.json")])
    assert "NO receipt" in out
    assert "REQ-INFRA-6830" in out


def test_a_dead_job_with_a_receipt_names_the_signal(tmp_path) -> None:
    import json

    receipt = tmp_path / "death.json"
    receipt.write_text(
        json.dumps(
            {"signal_name": "SIGTERM", "elapsed_s": 25200.0, "progress": {"cells": 38, "of": 39}}
        )
    )
    out = _module().render([("ab", 999_999_999, receipt)])
    assert "KILLED by SIGTERM" in out
    assert "38" in out


def test_a_live_job_is_reported_alive(tmp_path) -> None:
    out = _module().render([("self", os.getpid(), None)])
    assert "alive" in out


def test_the_header_carries_the_actual_date() -> None:
    """The report must state the date it was produced -- the drift this exists to stop."""
    out = _module().render([])
    assert datetime.now(UTC).strftime("%Y-%m-%d") in out
