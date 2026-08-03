"""REQ-ARC-WMTE-6091: the refactor prompt must be able to SHOW the engine it is refactoring.

BOTH DIRECTIONS, which is the point of the flag:
  * OFF reproduces the shipped prompt BYTE-IDENTICALLY -- including the defect. A fix that
    quietly changes the control arm makes the A/B it was written for uninterpretable.
  * ON delivers the engine's own substantive source lines into the RENDERED PROMPT STRING.
    Delivery is asserted on the rendered text, never on a value some dict made available:
    "availability is not delivery" is a documented failure mode of this codebase.

Every pattern here is proven under test by deletion (see `test_deleting_the_engine_block_is_
detected`): if the splice is removed from `refactor_prompt`, the ON assertions go red.
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.agentic.arc_executable_world_model import (
    Transition,
    WorldModelVerifier,
    refactor_prompt,
    refactor_show_engine_enabled,
)

ENGINE_SRC = """import numpy as np


def engine(grid, action, data):
    out = np.asarray(grid).copy()
    marker_row = int(out.shape[0]) - 1
    if action == 3:
        out[marker_row, 0] = 7
    return out


def is_level_complete(grid):
    return bool(np.asarray(grid)[0, 0] == 9)
"""

# The one line that carries THIS engine's logic and appears in no template anywhere.
SIGNATURE_LINE = "        out[marker_row, 0] = 7"


def _vr():
    """A real VerifyResult with real mismatches, produced by the shipped scorer."""
    g0 = np.zeros((3, 3), dtype=int)
    g1 = g0.copy()
    g1[1, 1] = 4
    g2 = g1.copy()
    g2[2, 2] = 5
    rows = [
        Transition(g0.copy(), 1, None, g1.copy(), 0, 0),
        Transition(g1.copy(), 2, None, g2.copy(), 0, 0),
    ]

    def wrong_engine(grid, action, data=None):
        return np.asarray(grid).copy()

    return WorldModelVerifier(rows, hud_mask=None).score(wrong_engine)


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("CARNOT_ARC_REFACTOR_SHOW_ENGINE", raising=False)
    assert refactor_show_engine_enabled() is False


def test_off_arm_is_byte_identical_to_the_shipped_prompt(monkeypatch):
    """The OFF arm must reproduce the shipped defect exactly: no engine, no passing cases."""
    monkeypatch.delenv("CARNOT_ARC_REFACTOR_SHOW_ENGINE", raising=False)
    vr = _vr()
    off = refactor_prompt("g", vr)
    # Explicitly "0" must render the same bytes as unset.
    monkeypatch.setenv("CARNOT_ARC_REFACTOR_SHOW_ENGINE", "0")
    assert refactor_prompt("g", vr) == off
    # The defect itself, asserted rather than described.
    assert SIGNATURE_LINE not in off
    assert "THE CURRENT ENGINE YOU ARE FIXING" not in off
    # Passing an engine_source has NO effect while the flag is off -- the flag, not the
    # argument, is what selects the arm.
    assert refactor_prompt("g", vr, engine_source=ENGINE_SRC) == off


def test_on_arm_delivers_engine_source_into_the_rendered_prompt(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_REFACTOR_SHOW_ENGINE", "1")
    assert refactor_show_engine_enabled() is True
    vr = _vr()
    on = refactor_prompt("g", vr, engine_source=ENGINE_SRC)
    assert "THE CURRENT ENGINE YOU ARE FIXING" in on
    assert SIGNATURE_LINE in on
    # The mismatch block survives the splice -- the engine is ADDED, nothing is displaced.
    assert "MISMATCHES:" in on
    assert "true_change" in on


def test_on_arm_reads_the_engine_off_disk_when_no_source_is_passed(monkeypatch, tmp_path):
    """The live refactor path passes no `engine_source`; it must still find the engine."""
    import carnot.agentic.arc_executable_world_model as mod

    store = tmp_path / "e3"
    (store / "g").mkdir(parents=True)
    (store / "g" / "world_model.py").write_text(ENGINE_SRC)
    monkeypatch.setattr(mod, "E3_DIR", store)
    monkeypatch.setenv("CARNOT_ARC_REFACTOR_SHOW_ENGINE", "1")
    on = refactor_prompt("g", _vr())
    assert SIGNATURE_LINE in on


def test_on_arm_degrades_to_the_off_prompt_when_the_engine_is_missing(monkeypatch, tmp_path):
    """A missing engine must not raise inside a live refinement round, and must not fabricate
    an empty code fence that reads as "the engine is empty"."""
    import carnot.agentic.arc_executable_world_model as mod

    monkeypatch.setattr(mod, "E3_DIR", tmp_path / "nonexistent")
    vr = _vr()
    monkeypatch.setenv("CARNOT_ARC_REFACTOR_SHOW_ENGINE", "1")
    on_missing = refactor_prompt("g", vr)
    monkeypatch.setenv("CARNOT_ARC_REFACTOR_SHOW_ENGINE", "0")
    assert on_missing == refactor_prompt("g", vr)


def test_oversize_engine_is_truncated_and_says_so(monkeypatch, tmp_path):
    """Truncation is announced and counted -- never silent censoring."""
    import carnot.agentic.arc_executable_world_model as mod

    store = tmp_path / "e3"
    (store / "g").mkdir(parents=True)
    big = ENGINE_SRC + ("\n# " + "x" * 100) * 400
    (store / "g" / "world_model.py").write_text(big)
    monkeypatch.setattr(mod, "E3_DIR", store)
    monkeypatch.setenv("CARNOT_ARC_REFACTOR_SHOW_ENGINE", "1")
    on = refactor_prompt("g", _vr())
    assert "further characters of this file were omitted" in on
    text, dropped = mod._current_engine_source("g")
    assert dropped > 0
    assert len(text) == mod._REFACTOR_ENGINE_SOURCE_MAX_CHARS


def test_deleting_the_engine_block_is_detected(monkeypatch):
    """MUTATION PROOF. Neutralise the splice the way a careless edit would -- make the source
    resolver return nothing -- and the ON arm must go red. A pattern whose removal leaves the
    suite green is decorative."""
    import carnot.agentic.arc_executable_world_model as mod

    monkeypatch.setenv("CARNOT_ARC_REFACTOR_SHOW_ENGINE", "1")
    monkeypatch.setattr(mod, "_current_engine_source", lambda game, **kw: ("", 0))
    on = refactor_prompt("g", _vr())
    with pytest.raises(AssertionError):
        assert SIGNATURE_LINE in on
