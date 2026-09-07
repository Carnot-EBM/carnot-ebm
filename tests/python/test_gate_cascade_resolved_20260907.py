"""A cascade whose upstream is retired has FIRED; reporting it as pending trains the
reader to discount the line.

REQ-CONDUCTOR-CASCADE-1, extended 2026-09-07. Observed 2026-09-06: the dashboard
reported `exp7081` as a pending cascade 42 minutes after exp7081 had been pre-emptively
skipped. The checker classified dependents by artifact presence, and a skipped task
writes no artifact, so it was indistinguishable from one that had not started.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import gate_cascade_check as gcc  # noqa: E402

SKIP_ROW = (
    "| 2026-09-06 15:31 UTC | Set-level entrance-bank sufficiency and conflict a | "
    "GATE_BLOCK | Pre-emptive skip: upstream retired (exp7080-recovered-entrance-bank) |"
)


def _log(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "conductor-log.md"
    p.write_text(text, encoding="utf-8")
    return p


def test_the_retired_upstream_id_is_read_from_the_detail(tmp_path: Path) -> None:
    # The row's SECOND column is the skipped dependent's title, truncated. The retired
    # upstream id lives in the detail; reading the title would join the wrong task.
    got = gcc.retired_upstreams(_log(tmp_path, SKIP_ROW + "\n"))
    assert got == {"exp7080-recovered-entrance-bank"}


def test_a_truncated_row_does_not_swallow_the_rest_of_the_log(tmp_path: Path) -> None:
    """The first draft used `\\(([^)]*)` over the whole file. A row whose detail is cut
    off has no closing paren, so the class ran across newlines and captured 928 "ids",
    one of them a 700-character blob. Parse per line, and only id-shaped tokens."""
    truncated = (
        "| 2026-09-06 15:31 UTC | A task | GATE_BLOCK | "
        "Pre-emptive skip: upstream retired (exp7080-cut-off-here"
    )
    text = truncated + "\n| 2026-09-06 15:32 UTC | Other | OK | 81 passed |\n"
    got = gcc.retired_upstreams(_log(tmp_path, text))
    assert got == {"exp7080-cut-off-here"}
    assert all(len(x) < 60 for x in got)


def test_several_retired_upstreams_on_one_row_are_all_captured(tmp_path: Path) -> None:
    row = (
        "| 2026-09-06 15:31 UTC | A task | GATE_BLOCK | Pre-emptive skip: upstream "
        "retired (exp7080-one, exp7081-two) |"
    )
    assert gcc.retired_upstreams(_log(tmp_path, row + "\n")) == {
        "exp7080-one",
        "exp7081-two",
    }


def test_an_ordinary_gate_block_is_not_a_retirement(tmp_path: Path) -> None:
    """Widening a matcher is how it starts matching everything. An ordinary GATE_BLOCK
    names an experiment id too, and must NOT be read as a retirement."""
    row = (
        "| 2026-09-06 07:48 UTC | Cold recomputation | GATE_BLOCK | 1 of 1 gate(s) "
        "failed; first failure: exp7065-three-family-entrance-proposal-ban |"
    )
    assert gcc.retired_upstreams(_log(tmp_path, row + "\n")) == set()


def test_a_missing_log_is_empty_not_an_error(tmp_path: Path) -> None:
    # Fail-open here is correct: no log means no evidence of retirement, so every
    # cascade stays reported. The failure direction is toward MORE alarm, not less.
    assert gcc.retired_upstreams(tmp_path / "absent.md") == set()


def test_an_id_before_the_marker_is_not_read_as_retired(tmp_path: Path) -> None:
    """The scan starts AFTER the marker. A task whose own TITLE carries an experiment id
    would otherwise be marked retired by its own row -- silencing a cascade that is still
    live. Found by mutation: replacing the slice with the whole line left every other
    test green, so this input is what makes the slice load-bearing rather than habit."""
    row = (
        "| 2026-09-06 15:31 UTC | exp7099 follow-up audit | GATE_BLOCK | "
        "Pre-emptive skip: upstream retired (exp7080-real-upstream) |"
    )
    got = gcc.retired_upstreams(_log(tmp_path, row + "\n"))
    assert got == {"exp7080-real-upstream"}
    assert "exp7099" not in got, "the row's own title must not mark itself retired"
