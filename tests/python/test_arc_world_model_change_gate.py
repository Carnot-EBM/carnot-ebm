"""REQ-ARC-WMTE-6010 / REQ-ARC-WMTE-6011 -- HUD-masked comparison + change-weighted trust gate.

These tests are HERMETIC on purpose: they build transitions in memory rather than calling
`collect_transitions`, so they run anywhere, with no `environment_files/` checkout, no game
sim, and no network. The REAL-corpus evidence -- the on-disk degenerate engines being
rejected and a hand-written correct engine being admitted -- lives in
`scripts/experiments/experiment_6011_world_model_change_gate_four_arm.py` and its artifact,
because that evidence needs the real games and cannot be asserted from a fixture.

What the fixtures encode is the STRUCTURE of the two defects, so that a regression in either
repair fails here immediately rather than at the next full survey:

  * `_hud_corpus` reproduces the REQ-6010 shape: a real mechanic (an avatar cell moving) plus
    a monotone step counter in the last row that ticks on EVERY transition. Full-grid exact
    match against a correct-but-HUD-blind engine is therefore 0 by construction, exactly as
    measured on the live path.
  * `_noop_heavy_corpus` reproduces the REQ-6011 / GAP-WM-TRUST-GATE shape: mostly no-ops with
    a minority of real changes, which is what lets an identity engine clear `accuracy >= 0.5`.
"""

from __future__ import annotations

import numpy as np
import pytest

# MODULE-LEVEL on purpose. Importing `arc_competition_agent` inside a test body charges its
# ~600MB import cost to that test's setup->teardown delta, which trips conftest's
# PytestMemoryWatchdog and produces a TEARDOWN ERROR -- an invisible failure. The fix is to
# pay the import at collection time (the same thing test_arc_early_stop.py already does), not
# to silence the watchdog with a `memory_watchdog_skip` marker.
from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.agentic.arc_executable_world_model import (
    Transition,
    WorldModelVerifier,
    apply_hud_mask,
    change_gate_decision,
    logical_hud_mask,
    world_model_change_gate_enabled,
    world_model_hud_mask_enabled,
    SUBMITTED_WORLD_MODEL_CHANGE_GATE_ENABLED,
    SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED,
    WORLD_MODEL_CHANGE_FIDELITY_THRESHOLD,
)

GRID_H, GRID_W = 8, 8
HUD_ROW = GRID_H - 1


def _hud_mask() -> np.ndarray:
    mask = np.zeros((GRID_H, GRID_W), dtype=bool)
    mask[HUD_ROW, :] = True
    return mask


def _hud_corpus(n: int = 6) -> list[Transition]:
    """A real mechanic (avatar steps right) plus a HUD counter that ticks every single step.

    CALIBRATION CORRECTED 2026-07-27 (REQ-ARC-WMTE-6014). This fixture previously redrew
    the ENTIRE 8-cell HUD row on every step against a 2-cell avatar, and justified that
    shape by citing su15 and lf52. Measuring mask coverage per game showed those two are
    not the typical case -- they are the PATHOLOGICAL case, and the reason the mask
    launders a zero-knowledge engine:

        game            changed cells inside the mask     changing transitions raw -> masked
        lf52                     1.0000                              60 -> 0
        su15                     0.7568                              28 -> 1
        every other game    0.0000 .. 0.2219 (s5i5 is the max)       moderate reduction

    On lf52 masking leaves the corpus with NO dynamics at all, under which the IDENTITY
    engine is a perfect model. So a fixture shaped like lf52 does not "reproduce the defect
    the mask exists to fix" -- it reproduces the defect the SWALLOW GUARD exists to refuse,
    and the guard now (correctly) refuses this shape.

    This fixture therefore uses the shape the other 15 games actually have: a counter that
    advances ONE cell per step against a multi-cell avatar. Exact match is still unattainable
    unmasked -- the engine cannot predict the counter, which is the whole point these tests
    assert -- but the mask no longer swallows the game. The su15/lf52 shape is kept and
    tested explicitly, as a case that MUST be refused, in
    tests/python/test_arc_hud_mask_coherence_and_swallow_guard.py.
    """

    out: list[Transition] = []
    for i in range(n):
        g0 = np.zeros((GRID_H, GRID_W), dtype=np.int16)
        g0[2:4, i : i + 2] = 5  # a 2x2 avatar
        g0[HUD_ROW, : i + 1] = 7  # a counter filled to i+1 cells
        g1 = np.zeros((GRID_H, GRID_W), dtype=np.int16)
        g1[2:4, i + 1 : i + 3] = 5  # avatar stepped right
        g1[HUD_ROW, : i + 2] = 7  # counter ticked by exactly ONE cell
        out.append(Transition(g0, 4, None, g1, 0, 0))
    return out


def _counter_only_corpus(n: int = 6) -> list[Transition]:
    """Transitions where ONLY the HUD counter ticks -- the game state is genuinely inert.

    These are the transitions the mask exists to reclassify. Unmasked they read as
    "changing" and the engine (which writes nothing on them, correctly) scores 0 fidelity on
    each; masked they become true no-ops and leave the fidelity average alone. That
    reclassification is where the mask's measured decision-level effect actually comes from
    -- see the dc22 numbers in `_hud_corpus`'s docstring.

    The avatar is placed against the right wall so `_avatar_engine`'s bounds check makes it
    correctly leave the grid alone, which is what an honest engine does on an inert action.
    """

    out: list[Transition] = []
    for i in range(n):
        g0 = np.zeros((GRID_H, GRID_W), dtype=np.int16)
        g0[2:4, GRID_W - 2 :] = 5  # avatar flush right -> a move would go out of bounds
        g0[HUD_ROW, : i + 1] = 7
        g1 = g0.copy()
        g1[HUD_ROW, : i + 2] = 7  # ONLY the counter ticks
        out.append(Transition(g0, 4, None, g1, 0, 0))
    return out


def _avatar_engine(grid, action, data):
    """Correct about the MECHANIC, blind to the HUD counter -- the realistic good engine."""

    g = np.asarray(grid).copy()
    cells = np.argwhere(g == 5)
    if len(cells) == 0:
        return g
    # Moves the WHOLE avatar footprint, not one cell. Updated 2026-07-27 alongside
    # `_hud_corpus`'s calibration fix, which made the avatar 2x2; a single-cell mover
    # against a 2x2 avatar is not "correct about the mechanic" and would make every test
    # here measure a broken engine instead of an honest partial one.
    y0, x0 = int(cells[:, 0].min()), int(cells[:, 1].min())
    y1, x1 = int(cells[:, 0].max()), int(cells[:, 1].max())
    if x1 + 1 < g.shape[1]:
        g[y0 : y1 + 1, x0 : x1 + 1] = 0
        g[y0 : y1 + 1, x0 + 1 : x1 + 2] = 5
    return g


def _identity_engine(grid, action, data):
    return np.asarray(grid)


def _noop_heavy_corpus(n_noop: int = 87, n_change: int = 33) -> list[Transition]:
    """The GAP-WM-TRUST-GATE shape: lp85's measured 87 no-ops to 33 changing transitions."""

    out: list[Transition] = []
    for _ in range(n_noop):
        g = np.zeros((4, 4), dtype=np.int16)
        out.append(Transition(g, 6, {"x": 0, "y": 0}, g.copy(), 0, 0))
    for i in range(n_change):
        g0 = np.zeros((4, 4), dtype=np.int16)
        # DISTINCT predecessor per changing transition. Without this the corpus is
        # ambiguous -- the same (state, action) maps to several different successors, and
        # NO deterministic engine can be correct on it, which would make the "is the pass
        # region reachable" probe below unfalsifiable for the wrong reason.
        g0[3, 3] = i + 1
        g1 = g0.copy()
        g1[i % 4, (i // 4) % 4] = 7
        out.append(Transition(g0, 6, {"x": 1, "y": 1}, g1, 0, 0))
    return out


def _perfect_on(corpus: list[Transition]):
    """An engine that is correct on this corpus by REPLAYING the observed successor.

    Used ONLY as the "must the pass region be reachable at all" probe. It is explicitly NOT
    the must-not-fire control -- a replay engine is a pass that could not have failed. The
    real must-not-fire control is the hand-written dc22 navigation engine measured on real
    transitions in experiment_6011.
    """

    table = {(t.grid.tobytes(), t.action): t.next_grid for t in corpus}

    def _engine(grid, action, data):
        return table.get((np.asarray(grid).tobytes(), action), np.asarray(grid))

    return _engine


# ---------------------------------------------------------------------------
# Flags ship OFF. If either of these ever fails, a default-off repair was flipped
# without the operator's decision.
# ---------------------------------------------------------------------------


def test_both_repairs_ship_default_off():
    assert SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED is False
    assert SUBMITTED_WORLD_MODEL_CHANGE_GATE_ENABLED is False


def test_flags_are_independent_arm_selectable(monkeypatch):
    """The four-arm matrix needs each repair selectable WITHOUT touching the other."""

    monkeypatch.delenv("CARNOT_ARC_WM_HUD_MASK", raising=False)
    monkeypatch.delenv("CARNOT_ARC_WM_CHANGE_GATE", raising=False)
    assert (world_model_hud_mask_enabled(), world_model_change_gate_enabled()) == (False, False)

    monkeypatch.setenv("CARNOT_ARC_WM_HUD_MASK", "1")
    assert (world_model_hud_mask_enabled(), world_model_change_gate_enabled()) == (True, False)

    monkeypatch.delenv("CARNOT_ARC_WM_HUD_MASK")
    monkeypatch.setenv("CARNOT_ARC_WM_CHANGE_GATE", "1")
    assert (world_model_hud_mask_enabled(), world_model_change_gate_enabled()) == (False, True)


# ---------------------------------------------------------------------------
# REQ-ARC-WMTE-6010 -- the HUD is inside the exact-match comparison
# ---------------------------------------------------------------------------


def test_hud_makes_exact_match_unattainable_and_the_mask_restores_it():
    """THE DEFECT AND THE REPAIR IN ONE ASSERTION.

    The same correct engine on the same transitions: 0.0 accuracy while the counter is in
    the comparison, 1.0 once it is masked out. This is the fixture form of the effect
    measured on real dc22 transitions (0.4583 -> 0.9167 whole-corpus, 4/4 seeds).
    """

    corpus = _hud_corpus()
    unmasked = WorldModelVerifier(corpus, hud_mask=None, hud_mask_enabled=False).score(
        _avatar_engine
    )
    assert unmasked.accuracy == 0.0, (
        "the fixture must reproduce the unattainable-by-construction shape"
    )
    assert unmasked.n_changing == len(corpus)

    masked = WorldModelVerifier(corpus, hud_mask=_hud_mask(), hud_mask_enabled=True).score(
        _avatar_engine
    )
    assert masked.accuracy == 1.0
    assert masked.change_accuracy == 1.0
    assert masked.change_fidelity == 1.0


def test_mask_never_silently_no_ops_every_outcome_is_named():
    """Requirement: "When no mask resolves, record that explicitly -- never a silent no-op."

    All four reachable statuses are asserted, including the one that matters most: a mask
    supplied at the WRONG RESOLUTION must report `shape_mismatch`, not quietly grade unmasked.
    """

    corpus = _hud_corpus()

    off = WorldModelVerifier(corpus, hud_mask=_hud_mask(), hud_mask_enabled=False).score(
        _avatar_engine
    )
    assert off.hud_mask_status == "disabled"
    assert off.hud_mask_cells == 0

    none_given = WorldModelVerifier(corpus, hud_mask=None, hud_mask_enabled=True).score(
        _avatar_engine
    )
    assert none_given.hud_mask_status == "unresolved"

    wrong_shape = np.zeros((3, 3), dtype=bool)
    wrong_shape[0, :] = True
    mismatched = WorldModelVerifier(corpus, hud_mask=wrong_shape, hud_mask_enabled=True).score(
        _avatar_engine
    )
    assert mismatched.hud_mask_status == "shape_mismatch"
    assert mismatched.hud_mask_cells == 0
    # and it must NOT have silently produced the masked answer
    assert mismatched.accuracy == 0.0

    applied = WorldModelVerifier(corpus, hud_mask=_hud_mask(), hud_mask_enabled=True).score(
        _avatar_engine
    )
    assert applied.hud_mask_status == "applied"
    assert applied.hud_mask_cells == GRID_W


def test_mask_status_is_a_fact_about_the_corpus_not_about_the_engine():
    """REGRESSION for a defect found reviewing this change: status must not depend on the engine.

    The first version derived `hud_mask_status` inside the scoring loop, so an engine that
    raised on every transition -- or an empty corpus -- would `continue` past the alignment
    check and report `unresolved` even though a perfectly good mask had been supplied. Whether
    a mask fits these grids is a fact about the MASK and the CORPUS; the engine has no say.
    """

    corpus = _hud_corpus()
    mask = _hud_mask()

    def _always_raises(grid, action, data):
        raise RuntimeError("this engine is broken")

    crashed = WorldModelVerifier(corpus, hud_mask=mask, hud_mask_enabled=True).score(_always_raises)
    assert crashed.n_correct == 0, "the engine really did fail every transition"
    assert crashed.hud_mask_status == "applied", "status must still reflect the supplied mask"
    assert crashed.hud_mask_cells == GRID_W

    # an empty corpus gets its own named status rather than being mislabelled "unresolved"
    empty = WorldModelVerifier([], hud_mask=mask, hud_mask_enabled=True).score(_avatar_engine)
    assert empty.hud_mask_status == "no_transitions"
    assert empty.hud_mask_cells == 0


def test_logical_hud_mask_uses_the_same_stride_as_to_logical():
    """A frame mask must be downsampled by `[::cell, ::cell]` -- the stride `to_logical` uses.

    Getting this wrong is the realistic way to produce a mask that is the right TYPE and the
    wrong CELLS, which no shape check would catch.
    """

    frame = np.zeros((8, 8), dtype=bool)
    frame[6, :] = True  # a HUD row at an even index, so cell=2 samples it
    out = logical_hud_mask(frame, 2)
    assert out is not None
    assert out.shape == (4, 4)
    expected = frame[::2, ::2]
    assert np.array_equal(out, expected)
    assert bool(out[3, :].all())

    # A row the stride never samples yields no logical mask at all -- reported as None so the
    # caller records "logical_downsample_empty" rather than pretending it masked something.
    odd = np.zeros((8, 8), dtype=bool)
    odd[7, :] = True
    assert logical_hud_mask(odd, 2) is None
    assert logical_hud_mask(None, 1) is None


def test_apply_hud_mask_refuses_a_wrong_shaped_mask_rather_than_broadcasting():
    grid = np.arange(16, dtype=np.int16).reshape(4, 4)
    good = np.zeros((4, 4), dtype=bool)
    good[3, :] = True
    assert np.array_equal(apply_hud_mask(grid, good)[3, :], np.zeros(4, dtype=np.int16))
    assert np.array_equal(apply_hud_mask(grid, good)[:3, :], grid[:3, :])
    # unchanged input (no in-place mutation of the caller's grid)
    assert grid[3, 0] == 12
    bad = np.zeros((2, 2), dtype=bool)
    assert np.array_equal(apply_hud_mask(grid, bad), grid)
    assert np.array_equal(apply_hud_mask(grid, None), grid)


# ---------------------------------------------------------------------------
# REQ-ARC-WMTE-6011 -- GAP-WM-TRUST-GATE
# ---------------------------------------------------------------------------


def test_identity_engine_clears_the_legacy_gate_and_the_new_gate_rejects_it():
    """The origin incident, as a fixture: the legacy gate's blind spot and the repair.

    This mirrors the gap file's measured lp85 number (identity scores 0.725 and PASSES
    `accuracy >= 0.5`). The real on-disk engines are checked in experiment_6011; this
    fixture is what fails fast if the repair regresses.
    """

    corpus = _noop_heavy_corpus()
    vr = WorldModelVerifier(corpus).score(_identity_engine)

    assert vr.accuracy == pytest.approx(87 / 120)
    assert vr.accuracy >= 0.5, "fixture must reproduce the legacy gate PASSING an identity engine"

    decision = change_gate_decision(vr, enabled=True)
    assert decision["legacy_accuracy_would_pass"] is True
    assert decision["passed"] is False
    assert decision["reason"] == "degenerate_engine_no_correct_changed_cells"
    assert decision["correct_changed_cells"] == 0
    assert decision["n_changing"] == 33


def test_gate_pass_region_is_not_empty():
    """MUST-NOT-FIRE: a gate that rejects everything is not an improvement.

    The probe engine here is a replay table, which is why this test asserts only that the
    pass region is REACHABLE -- the load-bearing must-not-fire evidence is the hand-written
    dc22 engine on real transitions in experiment_6011.
    """

    corpus = _noop_heavy_corpus()
    vr = WorldModelVerifier(corpus).score(_perfect_on(corpus))
    decision = change_gate_decision(vr, enabled=True)
    assert decision["passed"] is True
    assert decision["reason"] == "passed"
    assert decision["change_fidelity"] == 1.0
    assert decision["correct_changed_cells"] > 0


def test_change_fidelity_is_symmetric_where_cell_recall_is_blind():
    """THE REASON THIS GATE DOES NOT REUSE `cell_recall`.

    Two engines, identical on every cell reality changed, differing only by one write reality
    never made. `cell_recall` cannot tell them apart -- it masks to the true changes. The
    symmetric union fidelity must, and must charge the spurious write.
    """

    corpus = _hud_corpus()
    mask = _hud_mask()

    def _plus_spurious(grid, action, data):
        g = np.asarray(_avatar_engine(grid, action, data)).copy()
        g[0, 0] = 999
        return g

    clean = WorldModelVerifier(corpus, hud_mask=mask, hud_mask_enabled=True).score(_avatar_engine)
    dirty = WorldModelVerifier(corpus, hud_mask=mask, hud_mask_enabled=True).score(_plus_spurious)

    # the blindness, demonstrated rather than asserted in prose
    assert dirty.cell_recall == clean.cell_recall == 1.0
    assert dirty.correct_changed_cells == clean.correct_changed_cells
    # the fidelity metric sees it
    assert dirty.change_fidelity < clean.change_fidelity
    assert dirty.spurious_changed_cells == len(corpus)
    assert clean.spurious_changed_cells == 0


def test_nondegeneracy_floor_is_redundant_at_k1_and_independent_above_it():
    """The floor is NOT an independent gate condition at the default k=1 -- prove it both ways.

    Half one: at k=1 the floor cannot fire while the fidelity test passes. That is a theorem
    (zero correct changed cells forces the union score to 0), and it was confirmed over 924
    real measured arms in experiment_6011. Asserting it here stops anyone reading the two
    conditions as independent evidence when they are not.

    Half two: the floor DOES decide alone once k > 1. Without this half the channel would be
    structurally dead at every setting, which is the "877 stat blocks with an errors key and
    zero non-zero values" failure this project has already made once.
    """

    corpus = _hud_corpus()
    mask = _hud_mask()
    vr = WorldModelVerifier(corpus, hud_mask=mask, hud_mask_enabled=True).score(_avatar_engine)

    # a genuinely good engine: fidelity passes, and it has plenty of correct changed cells
    at_k1 = change_gate_decision(vr, enabled=True, min_correct_changed_cells=1)
    assert at_k1["passed"] is True
    assert at_k1["fidelity_ok"] is True
    assert at_k1["nondegenerate"] is True
    # 4 cells per transition: the 2x2 avatar's footprint. (Was 2 when the fixture had a
    # 1-cell avatar, before the 2026-07-27 calibration fix.)
    assert at_k1["correct_changed_cells"] == 4 * len(corpus)

    # half two: raise the floor above what this engine can supply and the floor alone rejects
    high = change_gate_decision(
        vr, enabled=True, min_correct_changed_cells=at_k1["correct_changed_cells"] + 1
    )
    assert high["fidelity_ok"] is True, "fidelity must be UNCHANGED -- only the floor moved"
    assert high["nondegenerate"] is False
    assert high["passed"] is False
    assert high["reason"] == "degenerate_engine_no_correct_changed_cells"

    # half one: the redundancy claim itself. No engine can present zero correct changed cells
    # AND a passing fidelity, so the (False, True) quadrant is empty by construction.
    degenerate = WorldModelVerifier(corpus, hud_mask=mask, hud_mask_enabled=True).score(
        _identity_engine
    )
    d = change_gate_decision(degenerate, enabled=True, min_correct_changed_cells=1)
    assert d["correct_changed_cells"] == 0
    assert d["change_fidelity"] == 0.0, "zero correct changed cells forces union fidelity to 0"
    assert d["fidelity_ok"] is False


def test_gate_catches_an_engine_that_hallucinates_changes_on_noop_transitions():
    """FOUND BY ATTACKING THIS GATE, NOT BY TESTING IT.

    `change_fidelity` scores GRID-CHANGING transitions only, which leaves it structurally
    blind to an engine that models every real change correctly AND ALSO invents a change on
    every NO-OP. Measured on real dc22 transitions before this channel existed: such an engine
    scored change_fidelity 0.7243 and PASSED, while its full-grid exact accuracy was 0.0000 --
    wrong about every single transition in the corpus. `plan_in_model` walks the engine
    forward, so it would hallucinate a transition at every step of every plan.

    Critically, the LEGACY accuracy gate CAUGHT this engine (0.0 < 0.5). Without this channel
    the repair would be strictly WORSE than the gate it replaces on this failure mode, which
    is the one outcome a trust-gate repair must never have.
    """

    # a corpus with BOTH populations, so the two conditions can be exercised independently
    corpus = _hud_corpus() + [
        Transition(
            np.full((GRID_H, GRID_W), 4, dtype=np.int16),
            6,
            {"x": 0, "y": 0},
            np.full((GRID_H, GRID_W), 4, dtype=np.int16),
            0,
            0,
        )
        for _ in range(8)
    ]
    mask = _hud_mask()

    def _hallucinator(grid, action, data):
        g = np.asarray(_avatar_engine(grid, action, data)).copy()
        g[0, 0] = 77  # a phantom write on EVERY call, no-ops included
        return g

    honest = WorldModelVerifier(corpus, hud_mask=mask, hud_mask_enabled=True).score(_avatar_engine)
    liar = WorldModelVerifier(corpus, hud_mask=mask, hud_mask_enabled=True).score(_hallucinator)

    # the no-op population is real and both engines were measured against it
    assert honest.n_noop == 8 == liar.n_noop
    assert honest.n_noop_hallucinated == 0
    assert honest.noop_hallucination_rate == 0.0
    assert liar.n_noop_hallucinated == 8
    assert liar.noop_hallucination_rate == 1.0

    # the blindness that motivated the channel: fidelity alone does NOT separate them enough
    assert liar.change_fidelity >= WORLD_MODEL_CHANGE_FIDELITY_THRESHOLD, (
        "if this ever fails the attack no longer reproduces and this test has become vacuous"
    )

    honest_decision = change_gate_decision(honest, enabled=True)
    liar_decision = change_gate_decision(liar, enabled=True)
    assert honest_decision["passed"] is True, "MUST-NOT-FIRE: the honest engine is still admitted"
    assert liar_decision["passed"] is False
    assert liar_decision["reason"] == "engine_hallucinates_changes_on_noop_transitions"
    # and the legacy gate would have caught it, which is why this channel is not optional
    assert liar_decision["legacy_accuracy_would_pass"] is False


def test_gate_refuses_a_corpus_with_no_changing_transitions():
    """ft09's real measured shape: 120 transitions, ZERO changing, legacy accuracy 1.0.

    An all-no-op corpus cannot distinguish a good engine from the identity engine, so a PASS
    there would be a pass that could not have failed. The gate refuses and says why.
    """

    corpus = [
        Transition(
            np.zeros((4, 4), dtype=np.int16), 6, None, np.zeros((4, 4), dtype=np.int16), 0, 0
        )
        for _ in range(20)
    ]
    vr = WorldModelVerifier(corpus).score(_identity_engine)
    assert vr.accuracy == 1.0, "an identity engine is PERFECT on an all-no-op corpus"
    decision = change_gate_decision(vr, enabled=True)
    assert decision["legacy_accuracy_would_pass"] is True
    assert decision["passed"] is False
    assert decision["reason"] == "no_changing_transitions"


def test_disabled_gate_still_emits_the_full_witness():
    """A control arm must record the same diagnostics as a treatment arm.

    Otherwise the four-arm comparison needs a re-run to get the control's numbers, and
    "the control was never measured" becomes indistinguishable from "the control was clean".
    """

    corpus = _noop_heavy_corpus()
    vr = WorldModelVerifier(corpus).score(_identity_engine)
    off = change_gate_decision(vr, enabled=False)
    on = change_gate_decision(vr, enabled=True)

    assert off["gate_enabled"] is False
    assert off["passed"] is True
    assert off["reason"] == "gate_disabled"
    # every measured field is identical between the arms; only the DECISION differs
    measured = (
        "change_fidelity",
        "correct_changed_cells",
        "spurious_changed_cells",
        "n_changing",
        "n_transitions",
        "legacy_accuracy",
        "change_accuracy",
        "cell_recall",
    )
    for key in measured:
        assert off[key] == on[key], key
    assert on["passed"] is False


def test_witness_fields_are_all_non_trivially_populated():
    """No None-valued diagnostics, no structurally dead channels.

    A census of this repo once found 877 stat blocks carrying an `errors` key with ZERO
    non-zero values anywhere. Every field this repair adds must be provably reachable with a
    non-trivial value, or it is decoration.
    """

    corpus = _hud_corpus()
    mask = _hud_mask()

    def _half_right(grid, action, data):
        g = np.asarray(_avatar_engine(grid, action, data)).copy()
        g[0, 0] = 999
        return g

    vr = WorldModelVerifier(corpus, hud_mask=mask, hud_mask_enabled=True).score(_half_right)
    decision = change_gate_decision(vr, enabled=True)
    for key, value in decision.items():
        assert value is not None, key
    # each numeric channel takes a value that is neither its zero nor its saturation point
    assert 0 < vr.n_changing == len(corpus)
    assert vr.correct_changed_cells > 0
    assert vr.spurious_changed_cells > 0
    assert 0.0 < vr.change_fidelity < 1.0
    assert vr.hud_mask_cells == GRID_W
    assert vr.hud_mask_status == "applied"


# ---------------------------------------------------------------------------
# The AGENT-SIDE resolver. Without these, the module-level repair could be perfect
# and still never reach the live path -- which is the failure mode this whole
# investigation exists to fix (the mask was live in the explorer and reached ZERO
# world-model comparators).
# ---------------------------------------------------------------------------


def _policy_with_explorer_mask(frame_mask, cell=2):
    policy = E3AgentPolicy.__new__(E3AgentPolicy)  # resolver only: no env, no game, no LLM
    policy.cell = cell
    explorer = type("_Explorer", (), {})()
    explorer.hud_mask = frame_mask
    policy.explorer = explorer
    return policy, explorer


def test_agent_resolver_reads_the_live_explorer_mask_and_names_every_failure(monkeypatch):
    frame_mask = np.zeros((8, 8), dtype=bool)
    frame_mask[6, :] = True
    policy, explorer = _policy_with_explorer_mask(frame_mask)

    monkeypatch.delenv("CARNOT_ARC_WM_HUD_MASK", raising=False)
    assert policy._world_model_hud_mask() == (None, "flag_disabled")

    monkeypatch.setenv("CARNOT_ARC_WM_HUD_MASK", "1")
    mask, reason = policy._world_model_hud_mask()
    assert reason == "resolved"
    assert mask.shape == (4, 4), "the frame mask must arrive DOWNSAMPLED to logical coordinates"
    assert int(mask.sum()) == 4

    # The explorer WIDENS its mask mid-run when Stage 2 admits the repair-added cells. The
    # resolver must read it at verification time, not cache it -- caching would pin the
    # pre-Stage-2 mask for the whole episode.
    explorer.hud_mask = np.ones((8, 8), dtype=bool)
    mask2, reason2 = policy._world_model_hud_mask()
    assert reason2 == "resolved"
    assert int(mask2.sum()) == 16, "a widened explorer mask must be picked up"

    explorer.hud_mask = None
    assert policy._world_model_hud_mask() == (None, "explorer_mask_unresolved")

    policy.explorer = None
    assert policy._world_model_hud_mask() == (None, "no_explorer")


def test_agent_resolver_reports_a_stride_emptied_mask_rather_than_a_bare_none(monkeypatch):
    """A frame mask on rows the logical stride never samples must be NAMED, not silently dropped."""

    monkeypatch.setenv("CARNOT_ARC_WM_HUD_MASK", "1")
    odd_row = np.zeros((8, 8), dtype=bool)
    odd_row[7, :] = True  # cell=2 samples rows 0,2,4,6 -- never row 7
    policy, _ = _policy_with_explorer_mask(odd_row, cell=2)
    assert policy._world_model_hud_mask() == (None, "logical_downsample_empty")


# ---------------------------------------------------------------------------
# The two repairs must not be entangled -- the four-arm matrix depends on it
# ---------------------------------------------------------------------------


def test_the_two_repairs_are_independently_observable():
    """Each flag must move its own number and leave the other arm's decision alone.

    If turning the mask on could not change the gate's verdict, the mask arm would be inert;
    if the gate's verdict were fixed by the mask, the two could not be attributed separately.
    Both directions are asserted here on ONE corpus.
    """

    # WITH COUNTER-ONLY TRANSITIONS (corrected 2026-07-27). The mask's decision-level
    # effect comes overwhelmingly from RECLASSIFYING counter-only transitions as no-ops,
    # not from cleaning up mixed ones: on the real dc22 corpus masking takes the changing
    # count from 89 to 54 and fidelity from 0.4694 to 0.8148. A fixture in which every
    # transition also moves the avatar has no counter-only transitions to reclassify, so
    # the mask cannot flip a verdict there and this test would assert a shape the real
    # corpus does not have.
    corpus = _hud_corpus() + _counter_only_corpus()
    mask = _hud_mask()

    arms = {}
    for mask_on in (False, True):
        vr = WorldModelVerifier(
            corpus, hud_mask=(mask if mask_on else None), hud_mask_enabled=mask_on
        ).score(_avatar_engine)
        for gate_on in (False, True):
            arms[(mask_on, gate_on)] = change_gate_decision(vr, enabled=gate_on)

    # gate OFF: passes in both mask arms (the shipped behaviour is untouched by the mask flag)
    assert arms[(False, False)]["passed"] is True
    assert arms[(True, False)]["passed"] is True
    # gate ON: the mask is what decides. Correct engine, HUD in the compare -> rejected.
    assert arms[(False, True)]["passed"] is False
    assert arms[(True, True)]["passed"] is True
    # and the mask flag moved the measured quantity, not just the verdict
    assert arms[(True, True)]["change_fidelity"] > arms[(False, True)]["change_fidelity"]
