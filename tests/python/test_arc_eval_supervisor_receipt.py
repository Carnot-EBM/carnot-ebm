"""Spec: REQ-ARC-WMTE-6640, SCENARIO-ARC-WMTE-6640-EVAL-ROW

The live-path eval emits a trajectory-supervisor receipt on every row.

INCIDENT 2026-09-01. `scripts/arc_leaderboard_eval.py` is the live-path harness and collected NO
supervisor evidence. `trajectory_supervisor_diagnostics()` had three callers and none could serve
a live applied run: `arc_scored_path_lever_harness.py` is frozen on the RETIRED Qwen3.5-9B-MTP pin
and refuses to start against the live Qwen3.8-27B; `experiment_6776` is hardcoded to shadow mode,
and a shadow receipt is not redirect evidence; `experiment_6558` only reduces receipts others
produce. A 13.8-hour live run that day returned two rows with `trajectory_supervisor` absent, and
exp6844's outcome-credit audit reported `supervisor_effect_eligible_score: 0`.

The field must be present on EVERY row, both paths. An absent field reads as zero to a flat
consumer, which cost a full day of wrong A/B reporting on 2026-08-21.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import arc_leaderboard_eval as ale  # noqa: E402


class _Policy:
    def __init__(self, payload: Any = None, raises: BaseException | None = None) -> None:
        self._payload = payload
        self._raises = raises

    def trajectory_supervisor_diagnostics(self) -> Any:
        if self._raises is not None:
            raise self._raises
        return self._payload


def test_an_applied_receipt_is_passed_through_unchanged() -> None:
    receipt = {
        "enabled": True,
        "mode": "applied",
        "arm_outcomes": {"reinduce": {"fired": 3, "helped": 1}},
        "stagnations_unredirected": 2,
    }
    assert ale.supervisor_row_field(_Policy(receipt)) == receipt


def test_a_disabled_supervisor_still_produces_a_field() -> None:
    """`{"enabled": False}` must be distinguishable from "the harness never looked"."""
    assert ale.supervisor_row_field(_Policy({"enabled": False})) == {"enabled": False}


def test_a_raising_diagnostics_call_becomes_an_error_marker() -> None:
    """Instrumentation must never take the row down; a 13-hour run must not die reporting itself."""
    out = ale.supervisor_row_field(_Policy(raises=RuntimeError("ledger exploded")))
    assert out == {"error": "RuntimeError:ledger exploded"}


def test_a_non_dict_diagnostics_value_becomes_an_error_marker() -> None:
    """A bare None would silently read as 'no firings' rather than 'instrumentation broke'."""
    assert ale.supervisor_row_field(_Policy(None)) == {"error": "non_dict_diagnostics:NoneType"}


def test_the_field_is_never_absent_for_any_policy_state() -> None:
    """The property that matters: every path yields a dict, so a consumer can always tell."""
    for policy in (
        _Policy({"enabled": True, "mode": "applied"}),
        _Policy({"enabled": False}),
        _Policy(raises=ValueError("x")),
        _Policy(None),
        _Policy([1, 2, 3]),
    ):
        assert isinstance(ale.supervisor_row_field(policy), dict)


def test_the_eval_row_carries_the_receipt() -> None:
    """The call-site test. A helper that nothing calls is the defect this incident WAS.

    The receipt was absent from live rows for months while the function existed, so proving the
    helper works is not enough -- the row itself must carry the key.
    """
    src = (REPO / "scripts" / "arc_leaderboard_eval.py").read_text()
    assert '"trajectory_supervisor": supervisor_row_field(policy),' in src
