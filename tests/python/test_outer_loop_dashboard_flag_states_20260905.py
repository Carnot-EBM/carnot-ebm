"""Spec: REQ-INFRA-6840, SCENARIO-INFRA-6840-D

Flag states are read by parsing the ledger, not by pattern-matching its text.

INCIDENT 2026-09-05. Two flags were recorded as measured nulls, which is what the ledger exists
to capture. `--record-null` writes an `evidence:` block into the flag's entry, ABOVE its
`state:` line.

The dashboard read states with a regex that captured from a flag's name to the next line
matching a bare `key:`, then searched that window for `state:`. `evidence:` is itself a bare
key, so the window closed before reaching `state`. Both flags returned "?" and fell out of BOTH
categories: the headline read "2/4 shipped-but-untested, 0 measured-null" while the ledger held
two off_measured entries.

Recording the finding is what made the finding invisible, and the more evidence a flag carried
the more certainly it vanished.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import outer_loop_dashboard as dash  # noqa: E402

_LEDGER = """flags:
  CARNOT_ARC_WITH_EVIDENCE:
    discovered: '2026-08-14'
    evidence:
    - date: '2026-09-05'
      evidence_paths:
      - path: results/arc_leaderboard_eval_runs/a.json
        sha256: abc123
      source: external
      verdict: 'EXTERNAL_MEASURED_NULL: fires and changes nothing.'
    last_measured: '2026-09-05'
    promotable: false
    state: off_measured
  CARNOT_ARC_PLAIN:
    discovered: '2026-08-14'
    promotable: false
    state: unevaluated
schema: carnot.arc.flag_ledger.v1
"""


@pytest.fixture()
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc_flag_ledger.yaml").write_text(_LEDGER)
    monkeypatch.setattr(dash, "REPO", tmp_path)
    return tmp_path


def test_a_flag_carrying_evidence_still_reports_its_state(ledger: Path) -> None:
    """The exact incident: an evidence block above state made the flag read '?'."""
    assert dash.flag_states(["CARNOT_ARC_WITH_EVIDENCE"]) == {
        "CARNOT_ARC_WITH_EVIDENCE": "off_measured"
    }


def test_a_flag_without_evidence_is_unaffected(ledger: Path) -> None:
    assert dash.flag_states(["CARNOT_ARC_PLAIN"]) == {"CARNOT_ARC_PLAIN": "unevaluated"}


def test_a_measured_null_reaches_the_measured_null_line(ledger: Path) -> None:
    """A flag must land in one category or the other, never neither."""
    lines = dash.flag_lines(dash.flag_states(["CARNOT_ARC_WITH_EVIDENCE", "CARNOT_ARC_PLAIN"]))
    assert "1/2 shipped-but-untested, 1 measured-null" in lines[0]
    assert any("MEASURED-NULL  CARNOT_ARC_WITH_EVIDENCE" in ln for ln in lines)


def test_an_unknown_flag_reports_a_question_not_an_answer(ledger: Path) -> None:
    """A wrong 'unevaluated' is an answer; '?' is a question. Prefer the question."""
    assert dash.flag_states(["CARNOT_ARC_NOT_IN_LEDGER"]) == {"CARNOT_ARC_NOT_IN_LEDGER": "?"}


def test_a_malformed_ledger_does_not_report_every_flag_as_untested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail visibly. Reporting 'unevaluated' for an unreadable ledger invents a coverage gap."""
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc_flag_ledger.yaml").write_text("flags: [this, is, a, list]\n")
    monkeypatch.setattr(dash, "REPO", tmp_path)
    assert dash.flag_states(["CARNOT_ARC_PLAIN"]) == {"CARNOT_ARC_PLAIN": "?"}
