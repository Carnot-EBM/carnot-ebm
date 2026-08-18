"""REQ-ARC-WMTE-6500: retrieval index + lean prompt, so prefill can stop carrying every transition.

WHY THIS EXISTS. The four original induction tools all VERIFY a candidate against evidence the
prompt already contained. None of them let the prompt stop containing it. Rendering every
transition is the largest single driver of prompt size, and prefill -- not decode -- is what the
K=4 concurrency probe timed out on at 17k prompt tokens, on a card where 25 game threads share 4
slots. `list_transitions` is the retrieval half: an index with no grids in it, so the model can
ask what evidence exists and fetch only what it needs.

Scenarios: SCENARIO-ARC-WMTE-6500-INDEX (the index reports every visible transition),
SCENARIO-ARC-WMTE-6500-NOLEAK (it never reveals the held-out tail),
SCENARIO-ARC-WMTE-6500-NOGRIDS (it returns no grid payload, or it rebuilds the prompt it
replaces), SCENARIO-ARC-WMTE-6500-DEFAULTOFF (unset env renders exactly as before),
SCENARIO-ARC-WMTE-6500-FLOOR (a zero/garbage budget cannot produce an exemplar-free prompt).
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.agentic.arc_induction_tool_loop import _lean_prompt_k
from carnot.agentic.arc_induction_tools import (
    TOOL_NAMES,
    InductionToolSession,
    dispatch_tool,
)


class _T:
    """Minimal transition double: the tools only read grid/next_grid/action."""

    def __init__(self, before, after, action=1):
        self.grid = np.asarray(before)
        self.next_grid = np.asarray(after)
        self.action = action


def _session(n=8):
    trans = []
    for i in range(n):
        b = np.zeros((6, 6), dtype=int)
        a = b.copy()
        a[i % 6, (i * 2) % 6] = i + 1  # a distinct changed cell per transition
        trans.append(_T(b, a, action=(i % 4) + 1))
    return InductionToolSession(trans, cell=1)


# SCENARIO-ARC-WMTE-6500-INDEX
def test_index_covers_every_visible_transition() -> None:
    s = _session()
    out = s.list_transitions()
    assert out["ok"] is True
    assert out["n_visible"] == len(s.visible)
    assert [r["t"] for r in out["transitions"]] == list(range(len(s.visible)))
    # The index must carry enough to TARGET a follow-up query, or it saves nothing.
    r = out["transitions"][0]
    assert r["changed_cells"] >= 1 and r["changed_bbox"] is not None and r["shape"] == [6, 6]


# SCENARIO-ARC-WMTE-6500-NOLEAK
def test_index_never_reveals_the_held_out_tail() -> None:
    """The tail is scored aggregate-only so a memorising engine shows a visible/held-out gap.
    An index over ALL transitions would leak the tail's shape and quietly defeat that split --
    the same invariant query_region already holds by indexing self.visible."""
    s = _session()
    assert len(s.held_out) > 0, "fixture must actually produce a tail, or this proves nothing"
    out = s.list_transitions()
    assert out["n_visible"] == len(s.visible)
    assert out["n_visible"] < len(s.transitions)
    assert max(r["t"] for r in out["transitions"]) == len(s.visible) - 1


# SCENARIO-ARC-WMTE-6500-NOGRIDS
def test_index_returns_no_grid_payload() -> None:
    """If the index shipped grids it would rebuild the prompt it exists to shrink."""
    out = _session().list_transitions()
    for r in out["transitions"]:
        assert set(r) == {"t", "action", "shape", "changed_cells", "changed_bbox"}


def test_index_is_reachable_through_dispatch_and_registered() -> None:
    s = _session()
    assert "list_transitions" in TOOL_NAMES, "unregistered tool is unreachable by the model"
    assert dispatch_tool(s, "list_transitions", "")["ok"] is True
    assert dispatch_tool(s, "list_transitions", "{}")["ok"] is True


def test_shape_change_is_reported_not_crashed() -> None:
    """A transition whose grid RESHAPES cannot have a cell-diff; report -1 rather than raise,
    because one odd transition must not cost the whole index."""
    s = InductionToolSession([_T(np.zeros((4, 4), int), np.zeros((5, 5), int))], cell=1)
    if s.visible:  # holdout_split may assign the single row either way
        assert s.list_transitions()["transitions"][0]["changed_cells"] == -1


# SCENARIO-ARC-WMTE-6500-DEFAULTOFF
def test_lean_prompt_is_off_by_default(monkeypatch) -> None:
    """Unset means the served prompt is byte-identical to before this feature existed."""
    monkeypatch.delenv("CARNOT_ARC_INDUCE_LEAN_PROMPT", raising=False)
    assert _lean_prompt_k() is None


def test_lean_prompt_reads_the_budget(monkeypatch) -> None:
    monkeypatch.setenv("CARNOT_ARC_INDUCE_LEAN_PROMPT", "2")
    assert _lean_prompt_k() == 2


# SCENARIO-ARC-WMTE-6500-FLOOR
@pytest.mark.parametrize("bad,expect", [("0", 1), ("-3", 1), ("garbage", None), ("", None)])
def test_zero_and_garbage_budgets(monkeypatch, bad: str, expect) -> None:
    """Zero floors to 1: a prompt with no worked example asks for an engine before the model has
    seen the grid encoding once. Garbage falls back to OFF rather than to a guess."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_LEAN_PROMPT", bad)
    assert _lean_prompt_k() == expect
