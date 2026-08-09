"""LEVER (REQ-ARC-WMTE-6241): induce-prompt enrichment -- semantic action names, explicit
changed-cell counts, and an object-identity cross-reference. Phase 3a of the 2026-08-08 ARC
live-agent improvement plan.

`CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT` is DEFAULT OFF pending a leave-one-game-out held-out
change-fidelity A/B (per the plan's own gate). When on, `induce_prompt` appends two additive
blocks that neither `_transitions_block` nor `objects_block` themselves are modified to produce:

  1. `_action_semantics_and_counts_block` -- semantic names (UP/DOWN/LEFT/RIGHT/SPACE/MOUSE for
     actions 1-6; action 7 is left as a bare integer, no established semantic meaning in this
     project) plus an explicit changed-cell COUNT, for the SAME sampled transitions
     `_transitions_block` already renders.
  2. `_object_identity_crossref_note` -- which object shape ids (per `object_hash`) are shared
     between `objects_block`'s two tables, computed rather than left as a text hint.

These tests mirror `test_object_perception_induction.py`'s structure and guard the same class of
regression: the injection sites are inside `induce_prompt`'s f-string, evaluated on every call, so
if a helper is ever removed the live path NameErrors -- the flag-on tests below catch that.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_executable_world_model import (
    Transition,
    _action_label,
    _n_changed_cells,
    _object_identity_crossref_note,
    induce_prompt,
    induce_prompt_enrichment_enabled,
)

BG = 5
AV = 9
GOAL = 14


def _grid(r: int, c: int) -> np.ndarray:
    g = np.full((6, 8), BG, dtype=np.int16)
    g[r, c] = AV
    g[r, c + 1] = AV
    g[r + 1, c] = AV  # 3-cell L-shaped avatar
    g[4, 6] = GOAL
    return g


def _trans():
    return [
        Transition(
            grid=_grid(1, 1),
            action=4,
            data=None,
            next_grid=_grid(1, 2),
            level_before=0,
            level_after=0,
        ),
        Transition(
            grid=_grid(1, 2),
            action=6,
            data={"x": 3, "y": 3},
            next_grid=_grid(1, 3),
            level_before=0,
            level_after=1,
        ),
    ]


def test_flag_defaults_off():
    assert induce_prompt_enrichment_enabled() is False


def test_flag_unset_byte_identical_to_before_this_lever(monkeypatch):
    """The load-bearing byte-identity property: with the flag off (the shipped default), the
    assembled prompt must contain neither new block, and must be identical to what
    `induce_prompt` produced before this lever existed (proven here by asserting the two
    specific marker strings are absent, matching test_object_perception_induction.py's own
    guard style for the sibling lever)."""
    monkeypatch.delenv("CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT", raising=False)
    p = induce_prompt("xx", _trans(), 1)
    assert "OBSERVED TRANSITIONS" in p
    assert "ACTION SEMANTICS AND CHANGE COUNTS" not in p
    assert "OBJECT IDENTITY CROSS-REFERENCE" not in p


def test_flag_explicit_off_never_crashes(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT", "0")
    p = induce_prompt("xx", _trans(), 1)
    assert "ACTION SEMANTICS AND CHANGE COUNTS" not in p
    assert "OBJECT IDENTITY CROSS-REFERENCE" not in p


def test_flag_on_appends_action_semantics_and_counts_block(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT", "1")
    p = induce_prompt("xx", _trans(), 1)
    assert "ACTION SEMANTICS AND CHANGE COUNTS" in p
    assert "ACTION4(RIGHT)" in p  # keyboard action from _trans()
    assert "ACTION6(MOUSE)" in p  # click action from _trans()
    assert "changed_cells=" in p


def test_flag_on_appends_object_identity_crossref(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT", "1")
    p = induce_prompt("xx", _trans(), 1)
    assert "OBJECT IDENTITY CROSS-REFERENCE" in p


def test_flag_on_without_object_perception_omits_crossref_note(monkeypatch):
    """The cross-ref note depends on objects_block's tables existing at all -- with object
    perception off, there is nothing to cross-reference, so the note must not appear even
    though the enrichment flag itself is on."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT", "1")
    monkeypatch.setenv("CARNOT_ARC_OBJECT_PERCEPTION", "0")
    p = induce_prompt("xx", _trans(), 1)
    assert "ACTION SEMANTICS AND CHANGE COUNTS" in p  # independent of object perception
    assert "OBJECT IDENTITY CROSS-REFERENCE" not in p


def test_flag_explicit_on_is_byte_identical_to_env_true(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT", "1")
    p_1 = induce_prompt("xx", _trans(), 1)
    monkeypatch.setenv("CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT", "true")
    p_true = induce_prompt("xx", _trans(), 1)
    assert p_1 == p_true


def test_action_label_semantic_names_1_through_6():
    assert _action_label(1) == "ACTION1(UP)"
    assert _action_label(2) == "ACTION2(DOWN)"
    assert _action_label(3) == "ACTION3(LEFT)"
    assert _action_label(4) == "ACTION4(RIGHT)"
    assert _action_label(5) == "ACTION5(SPACE)"
    assert _action_label(6) == "ACTION6(MOUSE)"


def test_action_label_action_7_and_reset_left_bare():
    """Action 7 has no established semantic meaning anywhere in this project's docs -- must
    NOT be guessed. RESET (0) is likewise never rendered as a semantic name here."""
    assert _action_label(7) == "ACTION7"
    assert _action_label(0) == "ACTION0"


def test_n_changed_cells_counts_and_handles_shape_mismatch():
    a = np.zeros((3, 3), dtype=np.int16)
    b = a.copy()
    b[0, 0] = 1
    b[1, 1] = 1
    assert _n_changed_cells(a, b) == 2
    assert _n_changed_cells(a, a) == 0
    mismatched = np.zeros((2, 2), dtype=np.int16)
    assert _n_changed_cells(a, mismatched) == a.size  # the FIRST arg's size, per the docstring


def test_object_identity_crossref_note_defensive_on_bad_input():
    """A degenerate transition list must not raise -- the note returns "" rather than break
    induction, matching objects_block's own defensive contract."""
    bad = [
        Transition(
            grid=np.zeros((0,), dtype=np.int16),
            action=0,
            data=None,
            next_grid=np.zeros((0,), dtype=np.int16),
            level_before=0,
            level_after=0,
        )
    ]
    assert _object_identity_crossref_note(bad) == ""
    assert _object_identity_crossref_note([]) == ""


def test_object_identity_crossref_note_finds_shared_shape():
    """The avatar in `_trans()` is the SAME 3-cell L-shape in both the INITIAL grid and the
    win-transition's pre-completion grid (only translated), so its shape id must be reported
    as shared."""
    note = _object_identity_crossref_note(_trans())
    assert "OBJECT IDENTITY CROSS-REFERENCE" in note
    assert "shape id(s) appear in BOTH object tables" in note
