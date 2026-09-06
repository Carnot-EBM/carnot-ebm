"""REQ-INFRA-6980: the dashboard derives its flag list from the ledger, never a name list.

INCIDENT 2026-09-06. `render` passed `flag_states` a hardcoded list of four names while
`ops/arc_flag_ledger.yaml` held 142. Two flags merged to main that morning were invisible the
same evening, and three of the five `off_measured` findings were hidden as well. A name
allowlist is one of the reader shapes this project records as failing silently while reporting
clean, and it survived inside the function repaired that same day to stop pattern-matching the
ledger: the parsing was fixed and the SELECTION was left enumerated.

The assertions below bite the RENDERED line, not a helper's return value. Three dashboard
defects this week computed correctly and read wrongly, so a test that only checks a return
value proves the wrong half.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import outer_loop_dashboard as dash  # noqa: E402

LEDGER = """\
flags:
  A_REACHABLE_UNTESTED:
    state: unevaluated
    benchmark_reachable: true
  B_UNREACHABLE_UNTESTED:
    state: unevaluated
    benchmark_reachable: false
  C_MEASURED_NULL:
    state: off_measured
  D_NO_REACHABLE_KEY:
    state: unevaluated
"""


@pytest.fixture()
def ledger(tmp_path: Path) -> Path:
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc_flag_ledger.yaml").write_text(LEDGER)
    return tmp_path


def test_reachable_untested_is_selected(ledger: Path) -> None:
    assert "A_REACHABLE_UNTESTED" in dash.dashboard_flag_names(ledger)


def test_every_measured_null_is_selected(ledger: Path) -> None:
    """A measured null is a reason to STOP spending; hiding one is worse than hiding an
    untested flag. Three of five were hidden before this change."""
    assert "C_MEASURED_NULL" in dash.dashboard_flag_names(ledger)


def test_unreachable_untested_is_not_listed(ledger: Path) -> None:
    names = dash.dashboard_flag_names(ledger)
    assert "B_UNREACHABLE_UNTESTED" not in names
    assert "D_NO_REACHABLE_KEY" not in names


def test_selection_is_not_a_name_list(ledger: Path) -> None:
    """The defect: a flag absent from a hardcoded list can never appear. A ledger-derived
    selection must surface a name the source code has never seen."""
    src = (REPO / "scripts" / "outer_loop_dashboard.py").read_text()
    assert "A_REACHABLE_UNTESTED" not in src
    assert "A_REACHABLE_UNTESTED" in dash.dashboard_flag_names(ledger)


def test_totals_count_the_whole_ledger(ledger: Path) -> None:
    total, unevaluated = dash.ledger_totals(ledger)
    assert total == 4
    assert unevaluated == 3


def test_the_RENDERED_headline_states_its_population() -> None:
    """Bites the rendered string. The other dashboard defects this week were lines that
    computed the right number and did not say what it meant."""
    line = dash.flag_lines({"X": "unevaluated", "Y": "off_measured"})[0]
    assert "reachable-untested" in line
    assert "measured-null" in line
    assert "in the ledger" in line
    assert "not benchmark-reachable" in line


def test_render_calls_the_derived_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bites the CALL SITE: the rule must reach `render`, not merely exist."""
    called: list[bool] = []

    def fake() -> list[str]:
        called.append(True)
        return []

    monkeypatch.setattr(dash, "dashboard_flag_names", fake)
    monkeypatch.setattr(dash, "flag_states", lambda names: {})
    with contextlib.suppress(Exception):
        dash.render([])
    assert called, "render() no longer derives its flag list from the ledger"
