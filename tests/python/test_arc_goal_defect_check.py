"""Tests for the OPT-IN goal defect check + evidence-carrying goal prompt (2026-08-01).

REQ-ARC-WMTE-6071 (goal defect rejection), REQ-ARC-WMTE-6072 (goal prompt transitions).

SPEC ENTRY OWED, AND SAID OUT LOUD RATHER THAN LEFT AS A SILENT GAP. These two REQ ids are
allocated above the current maximum (6070) but are NOT yet written into
`openspec/capabilities/arc-world-model-trust-energy/spec.md`, because that file was being
edited by a concurrent session throughout this work and a read-modify-write of it would have
risked clobbering another agent's uncommitted changes. `check_spec_coverage.py` only requires
that a test REFERENCE a requirement, so this passes the hook -- which is precisely why the debt
is recorded here in prose: a reference to an unwritten requirement would otherwise be
indistinguishable from a satisfied one. Write the two entries when the file is quiet.

THE TEST THAT MATTERS MOST IS THE INERTNESS ONE. Both knobs ship DEFAULT OFF so that the
control arm of the A/B is the SHIPPED path rather than a reimplementation of it. That claim is
worth exactly as much as its proof, so `test_flags_off_*` assert byte-identity of the prompt
and emptiness of the defect list with the env unset -- not merely "looks similar".

Every pattern here is proven under test by construction: each defect kind has a case that
FAILS if the kind is removed, which is the deletion check CLAUDE.md's Test-Run Record
Integrity Discipline requires of a guard's pattern list.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as e3


class _T:
    """Minimal Transition stand-in: the code under test only reads .grid/.next_grid."""

    def __init__(self, grid, next_grid):
        self.grid = np.asarray(grid)
        self.next_grid = np.asarray(next_grid)
        self.action = 1
        self.data = None
        self.level_before = 0
        self.level_after = 0


def _trans():
    a = np.zeros((4, 4), dtype=int)
    b = a.copy()
    b[0, 0] = 3
    c = b.copy()
    c[1, 1] = 5
    return [_T(a, b), _T(b, c)]


def _proposer():
    return e3.LocalGGUFProposer()


# --------------------------------------------------------------------------- inertness


def test_flags_off_goal_defects_is_empty(monkeypatch):
    """REQ-ARC-WMTE-6030: with the flag unset, the detector reports nothing at all --
    including on code it would certainly flag when on. This is the default-off guarantee."""
    monkeypatch.delenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", raising=False)
    code = "def is_level_complete(grid):\n    return False\n"
    assert _proposer()._goal_defects(code, _trans()) == []


def test_flags_off_goal_prompt_is_byte_identical(monkeypatch):
    """REQ-ARC-WMTE-6031: passing transitions with the flag unset must not move one byte of
    the prompt. If this fails, the control arm is not the shipped path and no A/B using it
    means anything."""
    monkeypatch.delenv("CARNOT_ARC_GOAL_PROMPT_TRANSITIONS", raising=False)
    p = _proposer()
    without = p._goal_only_prompt("tn36", None)
    with_trans = p._goal_only_prompt("tn36", None, _trans())
    assert without == with_trans


def test_flag_on_goal_prompt_carries_the_deltas(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_GOAL_PROMPT_TRANSITIONS", "1")
    p = _proposer()
    out = p._goal_only_prompt("tn36", None, _trans())
    assert "transitions YOU observed" in out
    assert len(out) > len(p._goal_only_prompt("tn36", None))


def test_flag_on_but_no_transitions_is_byte_identical(monkeypatch):
    """No evidence must not produce an empty evidence HEADER -- a header promising
    transitions with none under it is worse than no header."""
    monkeypatch.setenv("CARNOT_ARC_GOAL_PROMPT_TRANSITIONS", "1")
    p = _proposer()
    assert p._goal_only_prompt("tn36", None, None) == p._goal_only_prompt("tn36", None)
    assert p._goal_only_prompt("tn36", None, []) == p._goal_only_prompt("tn36", None)


@pytest.mark.parametrize("raw", ["0", "", "true", "yes", "2", "on", "True"])
def test_malformed_env_falls_back_to_off(monkeypatch, raw):
    """A typo'd env var must not change how the scored agent behaves."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", raw)
    monkeypatch.setenv("CARNOT_ARC_GOAL_PROMPT_TRANSITIONS", raw)
    assert e3._goal_defect_check_on() is False
    assert e3._goal_prompt_transitions_on() is False


@pytest.mark.parametrize("raw", [" 1", "1 ", "1"])
def test_whitespace_padded_one_is_ON_matching_the_sibling_flag(monkeypatch, raw):
    """DELIBERATE, and asserted so it cannot drift. `_reject_inert_engines` -- the sibling
    opt-in knob these two were modelled on -- reads `bool(raw) and raw.strip() == "1"`, so a
    shell that exports a padded value turns it ON. These flags follow that convention exactly.
    An earlier draft of this test asserted the opposite; the convention won, because two
    default-off induce knobs that disagree about what "1" means is a worse trap than either
    reading on its own."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", raw)
    monkeypatch.setenv("CARNOT_ARC_GOAL_PROMPT_TRANSITIONS", raw)
    assert e3._goal_defect_check_on() is True
    assert e3._goal_prompt_transitions_on() is True
    monkeypatch.setenv("CARNOT_ARC_INDUCE_REJECT_INERT", raw)
    assert e3._reject_inert_engines() is True


# --------------------------------------------------------------- each defect kind bites


def test_goal_constant_false_is_a_defect(monkeypatch):
    """A_DECLINED, the largest slice: 34 of 71 unplannable goals are `return False`."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    code = "def is_level_complete(grid):\n    return False\n"
    assert "goal_constant" in _proposer()._goal_defects(code, _trans())


def test_goal_constant_true_is_a_defect(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    code = "def is_level_complete(grid):\n    return True\n"
    assert "goal_constant" in _proposer()._goal_defects(code, _trans())


def test_uniformity_trope_is_caught_as_constant(monkeypatch):
    """C_UNIFORMITY (11 of 71). Not syntactically constant, but constant over every frame the
    agent has actually seen -- which is the property that makes it useless to the search."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    code = "import numpy as np\ndef is_level_complete(grid):\n    return bool(np.all(grid == 1))\n"
    assert "goal_constant" in _proposer()._goal_defects(code, _trans())


def test_missing_return_is_a_defect(monkeypatch):
    """D_NO_PREDICATE: a body that is a bare import. Detected on the AST, no execution."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    code = "def is_level_complete(grid):\n    import numpy as np\n"
    assert "goal_missing_return" in _proposer()._goal_defects(code, _trans())


def test_bare_return_counts_as_missing(monkeypatch):
    """`return` with no value yields None, which is falsy -- identical in effect to
    `return False`, so it must not read as a returning function."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    code = "def is_level_complete(grid):\n    return\n"
    assert "goal_missing_return" in _proposer()._goal_defects(code, _trans())


def test_raising_goal_is_a_defect(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    code = "def is_level_complete(grid):\n    return grid[99, 99] == 1\n"
    assert "goal_raises" in _proposer()._goal_defects(code, _trans())


def test_last_definition_wins(monkeypatch):
    """The shadowing mechanism the taxonomy identified: `_combine_world_model` concatenates an
    evidence-grounded goal and an evidence-free one, and PYTHON BINDS THE SECOND. The detector
    must judge the bound definition, not the first one it happens to walk past."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    code = (
        "import numpy as np\n"
        "def is_level_complete(grid):\n    return bool(grid[0, 0] == 3)\n"
        "def is_level_complete(grid):\n    return False\n"
    )
    assert "goal_constant" in _proposer()._goal_defects(code, _trans())


# ------------------------------------------------------------------- clean goals pass


def test_discriminating_goal_is_clean(monkeypatch):
    """The whole point: a predicate that separates the observed frames is NOT re-asked."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    code = "import numpy as np\ndef is_level_complete(grid):\n    return bool(grid[0, 0] == 3)\n"
    assert _proposer()._goal_defects(code, _trans()) == []


def test_no_transitions_still_reports_the_syntactic_defect(monkeypatch):
    """No evidence is not a clean bill: a body with no return is defective whether or not
    anything was observed. Only the RUNTIME probes go silent."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    code = "def is_level_complete(grid):\n    import numpy as np\n"
    assert _proposer()._goal_defects(code, None) == ["goal_missing_return"]
    assert _proposer()._goal_defects("def is_level_complete(grid):\n    return False\n", None) == []


def test_unparseable_and_absent_defer_to_the_existing_gates(monkeypatch):
    """`generate()` already owns syntax errors and a missing `def`. Reporting them again here
    would double-charge the re-ask budget for one fault."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    assert _proposer()._goal_defects("def is_level_complete(:\n", _trans()) == []
    assert _proposer()._goal_defects("def engine(g, a, d):\n    return g\n", _trans()) == []


# --------------------------------------------------------------------- budget separation


def test_goal_budget_is_independent_of_the_engine_budget(monkeypatch):
    """The confound guard. A shared counter would let the goal consume the engine's re-ask on
    ~89% of cells, so the treatment arm's ENGINES would silently get one fewer sample and the
    arm difference would be part goal-check and part engine-check-removal."""
    monkeypatch.delenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_REASKS", raising=False)
    monkeypatch.setenv("CARNOT_ARC_INDUCE_DEFECT_REASKS", "0")
    assert e3._induce_defect_reasks() == 0
    assert e3._goal_defect_reasks() == e3._GOAL_DEFECT_REASKS > 0


@pytest.mark.parametrize("raw,want", [("0", 0), ("3", 3), ("x", e3._GOAL_DEFECT_REASKS)])
def test_goal_reask_budget_env(monkeypatch, raw, want):
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_REASKS", raw)
    assert e3._goal_defect_reasks() == want


# ------------------------------------------------- the hang bound this flag introduces


def test_a_hanging_goal_is_bounded_and_accepted_not_flagged(monkeypatch):
    """THE EXPOSURE THIS FLAG CREATES. The shipped induce path never executes
    `is_level_complete`; with the flag on it does, so an unbounded loop in an induced predicate
    would hang induction where it previously could not. The watchdog must (a) return, and (b)
    return ACCEPT -- a probe that could not finish is an absence of evidence, not a defect.
    Reporting it as `goal_raises` would be a guard inventing a finding."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    monkeypatch.setattr(e3, "_GOAL_PROBE_TIMEOUT_S", 0.25)
    # The `return` is unreachable but PRESENT, so the syntactic check passes and the RUNTIME
    # probe is what has to terminate. An earlier draft used a body with no return at all, which
    # was correctly reported as `goal_missing_return` before the probe ever ran -- it tested the
    # AST check, not the watchdog.
    code = "def is_level_complete(grid):\n    while True:\n        pass\n    return False\n"
    t0 = time.time()
    out = _proposer()._goal_defects(code, _trans())
    assert time.time() - t0 < 10, "the watchdog did not bound the probe"
    assert out == [], "a probe that could not finish must accept, not fabricate a defect"


def test_timeout_is_not_reported_as_goal_raises(monkeypatch):
    """Named separately from the bound itself because it is the part that could regress
    silently: swallowing _GoalProbeTimeout into the per-grid `except Exception` would still
    terminate, but would relabel 'I could not check' as 'I checked and it raised'."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    monkeypatch.setattr(e3, "_GOAL_PROBE_TIMEOUT_S", 0.25)
    code = "def is_level_complete(grid):\n    while True:\n        pass\n    return False\n"
    assert "goal_raises" not in _proposer()._goal_defects(code, _trans())


def test_probe_grid_count_is_capped(monkeypatch):
    """The cap keeps the check's cost bounded and roughly constant across games -- wa30's
    window would otherwise probe 66 grids per candidate against cd82's 10. Constancy needs two
    distinct answers, not every frame, so the cap costs almost no detection power."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    # A predicate that FLIPS on its 7th call. Whether the probe sees the flip is decided purely
    # by the cap, so the two assertions below differ only in the cap -- that is what makes this
    # a behavioural proof the cap bites, rather than an assertion that happens to pass.
    code = "COUNT = [0]\ndef is_level_complete(grid):\n    COUNT[0] += 1\n    return COUNT[0] > 6\n"
    many = _trans() * 30  # 60 transitions -> 120 grids if nothing capped it

    monkeypatch.setattr(e3, "_GOAL_PROBE_MAX_GRIDS", 4)
    assert "goal_constant" in _proposer()._goal_defects(code, many), (
        "with a cap of 4 the probe cannot reach the 7th call, so it must see a constant"
    )

    monkeypatch.setattr(e3, "_GOAL_PROBE_MAX_GRIDS", 20)
    assert _proposer()._goal_defects(code, many) == [], (
        "with a cap of 20 the probe reaches the flip and must NOT report a constant"
    )


def test_cap_still_detects_a_discriminating_goal(monkeypatch):
    """The cap must not turn a genuinely discriminating predicate into a false 'constant'
    whenever the discriminating frame happens to sit past the cap. This asserts the pairing
    that makes that unlikely: grids are collected as (.grid, .next_grid) per transition, so
    even a cap of 2 sees a before/after pair rather than two halves of one state."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1")
    monkeypatch.setattr(e3, "_GOAL_PROBE_MAX_GRIDS", 2)
    code = "import numpy as np\ndef is_level_complete(grid):\n    return bool(grid[0, 0] == 3)\n"
    assert _proposer()._goal_defects(code, _trans()) == []


def test_witness_publishes_the_goal_reask_count():
    """0 with the flag on must be readable as "the gate never fired", which is impossible if
    the count is merged into the always-on engine counter."""
    p = _proposer()
    p.n_goal_defect_reasks = 4
    assert p.liveness_witness()["llm"]["goal_defect_reasks"] == 4
