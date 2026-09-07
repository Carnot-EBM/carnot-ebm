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


def test_a_parked_milestone_escalation_is_surfaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-INFRA-6840-D. The `[A-Z_]+` pattern dropped the escalation that means the
    loop has STOPPED. The conductor writes a park as `OPERATOR-ATTENTION: 2026.09.621
    parked` -- digit-leading, lowercase. A parked conductor is alive with no children and
    a frozen OK count, so the block alone cannot distinguish it from an idle one.
    """
    log = tmp_path / "ops" / "conductor-log.md"
    log.parent.mkdir(parents=True)
    log.write_text(
        "| 2026-09-07 10:00 UTC | OPERATOR-ATTENTION: 2026.09.621 parked | WARN | "
        "activation refused after 2 replans; edit roadmap-next to unpark |\n"
        "| 2026-09-07 10:05 UTC | OPERATOR-ATTENTION: WRONG_MODEL_LOADED | WARN | host |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dash, "REPO", tmp_path)
    kinds = dict(dash.attention_kinds("2026-09-07"))
    assert kinds["parked"] == 1, "the park escalation must reach the attention line"
    assert kinds["WRONG_MODEL_LOADED"] == 1, "uppercase kinds must still work"


def test_parks_of_different_milestones_group_under_one_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-INFRA-6840-D. Keying on the raw text would read as N distinct
    escalations, one per milestone, which is noise rather than a count."""
    log = tmp_path / "ops" / "conductor-log.md"
    log.parent.mkdir(parents=True)
    log.write_text(
        "| 2026-09-07 10:00 UTC | OPERATOR-ATTENTION: 2026.09.621 parked | WARN | a |\n"
        "| 2026-09-07 11:00 UTC | OPERATOR-ATTENTION: 2026.09.622 parked | WARN | b |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dash, "REPO", tmp_path)
    assert dict(dash.attention_kinds("2026-09-07")) == {"parked": 2}


def test_an_ordinary_row_is_still_not_an_escalation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-INFRA-6840-D. Widening a pattern is how a counter starts counting
    everything; assert the widening did not swallow normal rows."""
    log = tmp_path / "ops" / "conductor-log.md"
    log.parent.mkdir(parents=True)
    log.write_text(
        "| 2026-09-07 10:00 UTC | Some ordinary task | OK | 12 passed |\n"
        "| 2026-09-07 10:01 UTC | Plan next milestone | FAIL | Codex CLI error |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dash, "REPO", tmp_path)
    assert dash.attention_kinds("2026-09-07") == []
