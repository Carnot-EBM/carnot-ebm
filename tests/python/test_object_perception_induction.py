"""LEVER #1 (REQ-ARC-WMTE-5830): object-structured perception in the inducer prompt.

Behind CARNOT_ARC_OBJECT_PERCEPTION (default off), induce_prompt appends a connected-component object
table (translation-invariant object_hash + containment + adjacency) alongside the raw grid, so the LLM
inducer can track an object across frames after it moves -- which the order-1 raw-grid features cannot.

These tests also GUARD the default path: the injection is inside an f-string whose condition
`_object_perception_on()` is evaluated on EVERY induce_prompt call, so if that helper (or objects_block)
is ever removed, induce_prompt NameErrors on the live path -- test_induce_prompt_default_path_never_crashes
catches exactly that (it was a real conductor-commit-race regression, 2026-07-24).
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_executable_world_model import Transition, induce_prompt, objects_block

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
        Transition(grid=_grid(1, 1), action=4, data=None, next_grid=_grid(1, 2),
                   level_before=0, level_after=0),
        Transition(grid=_grid(1, 2), action=4, data=None, next_grid=_grid(1, 3),
                   level_before=0, level_after=0),
    ]


def test_induce_prompt_default_path_never_crashes(monkeypatch):
    # Flag unset -> the raw-grid-only prompt, no object block, NO crash (guards the race regression).
    monkeypatch.delenv("CARNOT_ARC_OBJECT_PERCEPTION", raising=False)
    p = induce_prompt("xx", _trans(), 1)
    assert "OBSERVED TRANSITIONS" in p
    assert "OBJECT STRUCTURE" not in p


def test_flag_on_appends_object_block(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_OBJECT_PERCEPTION", "1")
    p = induce_prompt("xx", _trans(), 1)
    assert "OBJECT STRUCTURE" in p
    assert "shape=" in p  # the translation-invariant object hash is present


def test_flag_off_is_byte_identical_to_unset(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_OBJECT_PERCEPTION", "0")
    p_off = induce_prompt("xx", _trans(), 1)
    monkeypatch.delenv("CARNOT_ARC_OBJECT_PERCEPTION", raising=False)
    p_unset = induce_prompt("xx", _trans(), 1)
    assert p_off == p_unset
    assert "OBJECT STRUCTURE" not in p_off


def test_objects_block_serializes_objects_and_shape_hash():
    block = objects_block(_trans())
    assert "INITIAL OBJECTS" in block
    assert "obj0" in block  # at least one object serialized
    assert "containment" in block and "adjacency" in block


def test_objects_block_defensive_on_bad_input():
    # A degenerate transition list must not raise -- objects_block returns "" rather than break induction.
    bad = [Transition(grid=np.zeros((0,), dtype=np.int16), action=0, data=None,
                      next_grid=np.zeros((0,), dtype=np.int16), level_before=0, level_after=0)]
    assert isinstance(objects_block(bad), str)  # no exception
