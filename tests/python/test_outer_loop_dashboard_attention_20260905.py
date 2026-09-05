"""Spec: REQ-INFRA-6840, SCENARIO-INFRA-6840-C

The dashboard surfaces the conductor's OPERATOR-ATTENTION escalations for today.

INCIDENT 2026-09-05. `scripts/conductor_run_sentinel.py` is report-only by design: its own
docstring says "WHAT IT NEVER DOES: kill anything. A false stop is worse than a slow human." Its
findings therefore need a human to act. They were written only to `ops/conductor-log.md`, which
the hourly check does not read.

At 00:04Z the sentinel named an orphaned `llama-server` holding 21.9 GB across both GPUs on the
conductor's own port. At 00:15Z the same orphan was found and killed by hand, re-derived from
nothing, while the escalation naming it sat in the log. A detection nobody reads is a detection
that did not happen.

The guard fired 12 times before this and was acted on once, by accident.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import outer_loop_dashboard as dash  # noqa: E402

_LOG = """| 2026-09-05 00:04 UTC | OPERATOR-ATTENTION: ORPHANED_LLAMA_SERVER | WARN | host: pid 1 |
| 2026-09-05 00:04 UTC | OPERATOR-ATTENTION: AUDIT_FINDING_UNTRIAGED | WARN | a.py OPEN 1 days |
| 2026-09-05 00:05 UTC | OPERATOR-ATTENTION: AUDIT_FINDING_UNTRIAGED | WARN | b.py OPEN 1 days |
| 2026-09-04 23:00 UTC | OPERATOR-ATTENTION: ORPHANED_LLAMA_SERVER | WARN | yesterday, not today |
| 2026-09-05 00:06 UTC | Some ordinary task | OK | 12 passed |
"""


@pytest.fixture()
def logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point REPO at tmp_path so the test never reads the real conductor log.

    A test that reads `ops/conductor-log.md` would pass or fail on whatever the conductor
    happened to be doing, which is not a test.
    """
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "conductor-log.md").write_text(_LOG)
    monkeypatch.setattr(dash, "REPO", tmp_path)
    return tmp_path


def test_escalations_are_counted_by_kind(logged: Path) -> None:
    assert dash.attention_kinds("2026-09-05") == [
        ("AUDIT_FINDING_UNTRIAGED", 2),
        ("ORPHANED_LLAMA_SERVER", 1),
    ]


def test_yesterdays_escalation_is_not_counted_today(logged: Path) -> None:
    """The orphan warning repeats daily; a stale count would read as a fresh one."""
    kinds = dict(dash.attention_kinds("2026-09-05"))
    assert kinds["ORPHANED_LLAMA_SERVER"] == 1


def test_an_ordinary_row_is_not_an_escalation(logged: Path) -> None:
    assert all(k != "Some ordinary task" for k, _ in dash.attention_kinds("2026-09-05"))


def test_a_quiet_day_produces_no_line(logged: Path) -> None:
    assert dash.attention_kinds("2026-09-01") == []
    assert dash.attention_line([]) == ""


def test_the_line_names_every_kind_with_its_count() -> None:
    line = dash.attention_line([("AUDIT_FINDING_UNTRIAGED", 17), ("ORPHANED_LLAMA_SERVER", 1)])
    assert "ORPHANED_LLAMA_SERVER=1" in line
    assert "AUDIT_FINDING_UNTRIAGED=17" in line


def test_a_rare_kind_is_not_hidden_behind_a_common_one() -> None:
    """The orphan warning is 1 of 18 rows. Truncating to the top kind would have hidden it."""
    line = dash.attention_line([("AUDIT_FINDING_UNTRIAGED", 17), ("ORPHANED_LLAMA_SERVER", 1)])
    assert "ORPHANED_LLAMA_SERVER" in line
