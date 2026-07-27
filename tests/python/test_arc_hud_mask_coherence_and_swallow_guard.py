"""REQ-ARC-WMTE-6010/6014/6015 corrigendum tests (2026-07-27 adversarial review).

Every test here reproduces a SPECIFIC defect that shipped, not a synthetic happy path.
The project's own recorded failure mode is "a guard that does not fire on its own origin
incident", so each test names the incident it locks down.

The four defects:

  1. MASK COHERENCE (REQ-6010 corrigendum). `WorldModelVerifier` discarded a supplied mask
     unless a module flag was also set, while `score_change_weighted_consistency` applied
     the same mask unconditionally. Two comparators, two conventions, inside ONE
     `select_trusted_world_model` decision -- whose docstring promises they "must move
     TOGETHER". Measured consequence in exp6013: `change_fidelity` differed between the
     mask=0 and mask=1 arms on 0 of 162 paired arms (the gate was NEVER masked) while
     `incumbent_consistency` differed on 9 of 162.

  2. SWALLOW GUARD (REQ-6014). On lf52 the explorer's HUD classifier selects cells where
     the GAME STATE lives: 100% of truly-changed cells fall inside the mask and masking
     leaves 0 of 60 changing transitions. A dynamics-free corpus makes the IDENTITY engine
     optimal, which is how a "better measurement" launders a zero-knowledge engine into
     admission.

  3. PROMPT COHERENCE (REQ-6010). `_transitions_block` classified transitions
     changing-vs-inert on RAW grids while the verifier graded on MASKED grids, so the
     prompt could assert "this game has no inert actions" about a corpus the grader had
     already decided was almost entirely inert.

  4. ORIGIN-FIXTURE DRIFT (REQ-6015). The engines cited as the GAP-WM-TRUST-GATE origin
     incident live in a store that ANY induction run rewrites in place, and one did --
     ft09's 12-bare-`return grid` engine was replaced within hours of the artifact citing
     it. A guard asserted against a mutable store stops testing its own origin incident.
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_world_model_trust_energy as te


# --------------------------------------------------------------------------- helpers
def _t(grid, nxt, action=1):
    return e3.Transition(
        grid=np.asarray(grid),
        action=action,
        next_grid=np.asarray(nxt),
        data=None,
        level_before=0,
        level_after=0,
    )


def _corpus_with_hud(n_changing=6, n_noop=6, size=16):
    """A corpus whose LAST ROW is a monotone counter (the HUD) that ticks one cell per step.

    Constructed, not collected, so the test is deterministic and needs no game files. The
    counter makes EVERY transition differ in raw comparison, which is exactly the condition
    REQ-6010 exists to handle: raw no-ops are structurally impossible.

    PROPORTIONS MATTER AND ARE CALIBRATED TO THE REAL CORPUS. The first version of this
    fixture used an 8x8 grid, a full-width HUD row rewritten every step, and a 1-cell
    avatar. That gives a changed-cell overlap ABOVE 0.5, so REQ-6014's swallow guard
    correctly refused the mask and the must-not-fire test failed -- on the fixture, not on
    the code. Real games are 64x64 with a 64-cell HUD and multi-cell moving objects, and
    measure 0.0000-0.2219 overlap. A 3x3 avatar on a 16x16 grid with a one-cell-per-step
    counter reproduces that regime, which is what makes the must-not-fire assertion mean
    something rather than merely pass.
    """
    # INTERLEAVED, not blocked. `select_trusted_world_model` grades on the last third of
    # the list, so a fixture with all its changing transitions up front hands the gate a
    # tail with n_changing == 0 -- under which change_fidelity is 0.0 for every engine and
    # the arms are indistinguishable for a reason that has nothing to do with the mask.
    order = []
    for i in range(max(n_changing, n_noop)):
        if i < n_changing:
            order.append(True)
        if i < n_noop:
            order.append(False)
    rows = []
    for i, is_change in enumerate(order):
        g = np.zeros((size, size), dtype=int)
        y = 1 if is_change else 5
        g[y : y + 3, 1:4] = 5  # a 3x3 "avatar"
        g[size - 1, : (i % (size - 1)) + 1] = 9  # the HUD counter, filled so far
        nxt = g.copy()
        if is_change:
            nxt[y : y + 3, 1:4] = 0
            nxt[y : y + 3, 5:8] = 5  # a REAL multi-cell state change
        nxt[size - 1, ((i + 1) % (size - 1)) + 1 - 1] = 9  # the counter ticks ONE cell
        rows.append(_t(g, nxt))
    mask = np.zeros((size, size), dtype=bool)
    mask[size - 1, :] = True  # the HUD row only
    return rows, mask


def _identity(grid, action, data):
    return grid


def _perfect(grid, action, data):
    """Models the avatar move but NOT the HUD tick -- the honest PARTIAL engine a graded
    gate exists to admit. Modelling the counter would require state the engine never sees,
    which is the whole reason the HUD has to leave the comparison."""
    g = np.asarray(grid).copy()
    cells = np.argwhere(g[:-1, :] == 5)
    if len(cells) == 0:
        return g
    y0, x0 = int(cells[:, 0].min()), int(cells[:, 1].min())
    if x0 != 1:
        return g
    g[y0 : y0 + 3, 1:4] = 0
    g[y0 : y0 + 3, 5:8] = 5
    return g


# ------------------------------------------------- DEFECT 1: mask coherence
def test_supplying_a_mask_without_enabling_it_is_recorded_not_silently_dropped():
    """The exp6013 failure state must be VISIBLE, not silent.

    Before the fix a caller could pass a real mask, get no masking at all, and the only
    trace was `hud_mask_status="disabled"` -- which looks identical to "no mask was
    supplied". `hud_mask_silently_dropped` separates the two.
    """
    rows, mask = _corpus_with_hud()
    cands = [te.WorldModelCandidate(name="c", engine=_identity, is_level_complete=lambda g: False)]

    dropped = te.select_trusted_world_model(
        rows, cands, hidden_state=True, hud_mask=mask, hud_mask_enabled=False
    )
    assert dropped.hud_mask_supplied is True
    assert dropped.hud_mask_enabled is False
    assert dropped.hud_mask_silently_dropped is True, (
        "a supplied-but-disabled mask must be reportable; this is the state in which "
        "exp6013 measured mask-off twice and called it 'both mask settings'"
    )

    applied = te.select_trusted_world_model(
        rows, cands, hidden_state=True, hud_mask=mask, hud_mask_enabled=True
    )
    assert applied.hud_mask_enabled is True
    assert applied.hud_mask_silently_dropped is False


def test_score_change_weighted_consistency_honours_the_flag_not_just_the_argument():
    """The SPLIT CONVENTION itself, tested where it lived.

    This function used to call `apply_hud_mask` unconditionally on whatever mask it was
    handed, while `WorldModelVerifier` -- consuming the SAME mask in the SAME decision --
    consulted the module flag and discarded it. This asserts the two now agree at the level
    of the individual function, not merely because a caller above them nulls the mask first.
    A test that only exercised `select_trusted_world_model` would keep passing if this
    function regressed, because the caller's own `_effective_mask` masks the bug.
    """
    rows, mask = _corpus_with_hud()

    disabled = te.score_change_weighted_consistency(
        rows, _perfect, hud_mask=mask, hud_mask_enabled=False
    )
    none_supplied = te.score_change_weighted_consistency(rows, _perfect, hud_mask=None)
    assert disabled.consistency == none_supplied.consistency, (
        "flag OFF + mask supplied must equal no mask at all; before the fix this function "
        "masked regardless and disagreed with WorldModelVerifier inside one decision"
    )

    enabled = te.score_change_weighted_consistency(
        rows, _perfect, hud_mask=mask, hud_mask_enabled=True
    )
    assert enabled.consistency != disabled.consistency, (
        "flag ON must actually mask, or the parameter is decorative"
    )


def test_every_comparator_masks_together_or_not_at_all():
    """`select_trusted_world_model`'s docstring promise, asserted instead of asserted-in-prose.

    With masking OFF, BOTH the incumbent consistency and the change-gate quantity must be
    computed on unmasked grids. Before the fix the first was masked and the second was not,
    so a single decision mixed two notions of "the same state".
    """
    rows, mask = _corpus_with_hud()
    cands = [te.WorldModelCandidate(name="c", engine=_perfect, is_level_complete=lambda g: False)]

    off_nomask = te.select_trusted_world_model(rows, cands, hidden_state=True, hud_mask=None)
    off_withmask = te.select_trusted_world_model(
        rows, cands, hidden_state=True, hud_mask=mask, hud_mask_enabled=False
    )
    a, b = off_nomask.selected_score, off_withmask.selected_score
    assert a.heldout_change_consistency == b.heldout_change_consistency, (
        "with the flag OFF, supplying a mask must change NOTHING. Before the fix "
        "score_change_weighted_consistency masked unconditionally and this differed."
    )
    assert a.change_gate.get("change_fidelity") == b.change_gate.get("change_fidelity")

    on = te.select_trusted_world_model(
        rows, cands, hidden_state=True, hud_mask=mask, hud_mask_enabled=True
    )
    # And with the flag ON the mask must actually reach the GATE quantity -- the exact
    # thing that did not happen on any of exp6013's 162 arms.
    assert on.selected_score.change_gate.get("hud_mask_status") == "applied"
    assert on.selected_score.change_gate.get("change_fidelity") != a.change_gate.get(
        "change_fidelity"
    ), "the mask must move the gate quantity, or the mask arm is a no-op"


# ------------------------------------------------- DEFECT 2: the swallow guard
def test_swallow_guard_fires_when_the_mask_deletes_the_games_dynamics():
    """MUST-FIRE, against the lf52 shape: a mask covering the cells that actually change.

    lf52's real numbers: changed-cell overlap 1.0, 60 changing transitions -> 0 after
    masking. Reproduced here as a constructed corpus so the assertion cannot drift when a
    game file changes.
    """
    rows, _ = _corpus_with_hud(n_changing=6, n_noop=6, size=16)
    # A mask over the PLAY AREA (the avatar's rows) as well as the HUD -- the lf52
    # pathology, where the "HUD" classifier selects cells the game actually uses.
    bad = np.zeros((16, 16), dtype=bool)
    bad[1:4, :] = True
    bad[5:8, :] = True
    bad[15, :] = True

    rec = e3.hud_mask_swallow_check(rows, bad)
    assert rec["checked"] is True
    assert rec["swallows"] is True
    assert rec["reason"] in (
        "mask_removes_all_dynamics",
        "mask_overlaps_majority_of_changed_cells",
        "no_changed_cells_outside_mask_cannot_distinguish",
    )

    v = e3.WorldModelVerifier(rows, hud_mask=bad, hud_mask_enabled=True)
    assert v.hud_mask_status == "refused_swallows_dynamics"
    assert v.hud_mask is None, "a swallowing mask must never reach the comparison"
    vr = v.score(_identity)
    assert e3.change_gate_decision(vr, enabled=True)["hud_mask_swallow_guard_fired"] is True


def test_swallow_guard_does_NOT_fire_on_an_honest_hud_mask():
    """MUST-NOT-FIRE. A guard that refuses every mask is not an improvement over no guard.

    The honest games measured 0.0000-0.2219 changed-cell overlap; the threshold is 0.5, the
    midpoint of the only wide gap in the measured distribution (0.2219 -> 0.7568).
    """
    rows, mask = _corpus_with_hud()
    rec = e3.hud_mask_swallow_check(rows, mask)
    assert rec["checked"] is True
    assert rec["swallows"] is False, "an honest HUD-row mask must be applied, not refused"
    assert rec["changed_cell_overlap"] < e3.HUD_MASK_MAX_CHANGED_CELL_OVERLAP

    v = e3.WorldModelVerifier(rows, hud_mask=mask, hud_mask_enabled=True)
    assert v.hud_mask_status == "requested"
    assert v.score(_identity).hud_mask_status == "applied"


def test_swallow_check_distinguishes_a_bad_mask_from_an_inert_corpus():
    """A refusal must say WHICH of the two indistinguishable situations it is refusing on.

    Collapsing "the mask covers the game" and "this corpus has no state change, so the only
    cells that moved were the counter's" into one reason is the same clean-vs-unmeasurable
    conflation `noop_ok_is_vacuous` exists to prevent. Both are refused (over-masking
    destroys correctness, so refusing is the safe direction) but they are reported apart.
    """
    inert, mask = _corpus_with_hud(n_changing=0, n_noop=6)
    rec = e3.hud_mask_swallow_check(inert, mask)
    assert rec["swallows"] is True
    assert rec["reason"] == "no_changed_cells_outside_mask_cannot_distinguish"
    assert rec["n_changed_cells_inside_mask"] == rec["n_changed_cells_total"]

    # And a corpus with NO transitions at all is neither -- it is unmeasurable.
    empty = e3.hud_mask_swallow_check([], mask)
    assert empty["swallows"] is False
    assert empty["reason"] == "no_transitions"


# ------------------------------------------------- DEFECT 3: prompt coherence
def test_prompt_classifies_transitions_by_the_same_rule_the_verifier_grades_by():
    """The prompt must not assert "no inert actions" about a corpus the grader calls inert.

    With a ticking HUD every transition differs raw, so the unmasked prompt sees ZERO
    no-ops and shows none. The masked prompt must recover them.
    """
    rows, mask = _corpus_with_hud(n_changing=4, n_noop=6)
    raw_noops = [t for t in rows if np.array_equal(t.grid, t.next_grid)]
    assert raw_noops == [], "fixture precondition: the HUD makes raw no-ops impossible"

    unmasked = e3._transitions_block(rows, k=8, hud_mask=mask, hud_mask_enabled=False)
    masked = e3._transitions_block(rows, k=8, hud_mask=mask, hud_mask_enabled=True)
    assert unmasked != masked, "masking must change WHICH transitions the LLM is shown"
    # The masked block draws from a genuinely non-empty no-op pool; the unmasked one cannot.
    n_masked_noop = sum(
        1
        for t in rows
        if np.array_equal(e3.apply_hud_mask(t.grid, mask), e3.apply_hud_mask(t.next_grid, mask))
    )
    assert n_masked_noop > 0


def test_prompt_masking_is_off_by_default():
    """Default-off discipline: with the flag unset the prompt must be byte-identical."""
    rows, mask = _corpus_with_hud()
    assert e3._transitions_block(rows, k=8) == e3._transitions_block(
        rows, k=8, hud_mask=mask, hud_mask_enabled=False
    )


# ------------------------------------------------- DEFECT 4: origin fixture drift
def test_origin_fixture_engines_are_the_degenerate_ones_the_gap_entry_names():
    """The frozen ft09 engine must still be the 12-bare-`return grid` identity engine.

    The MUTABLE store no longer contains it -- a live run replaced it with a 2-branch
    mutating engine on 2026-07-27. This test reads the frozen copy, so the origin-incident
    claim in ops/verifier_gaps.md stays checkable regardless of what runs next.
    """
    src = (e3.E3_ORIGIN_FIXTURES_DIR / "ft09" / "world_model.py").read_text()
    assert src.count("return grid") >= 12, (
        "the frozen ft09 fixture is no longer the origin-incident identity engine; "
        "it must never be overwritten (see REQ-ARC-WMTE-6015)"
    )


@pytest.mark.parametrize("game", ["ft09", "lp85"])
def test_change_gate_rejects_the_frozen_origin_engines(game):
    """The catch, asserted against the FROZEN engines rather than whatever is on disk.

    Uses a constructed corpus shaped like each game's real one so the test needs no game
    files: the point is that these ENGINES are degenerate, which is a property of the
    engine source, not of the corpus.
    """
    engine, _ = e3.load_origin_fixture_engine(game)
    rows, _mask = _corpus_with_hud(n_changing=6, n_noop=12)
    vr = e3.WorldModelVerifier(rows).score(engine)
    decision = e3.change_gate_decision(vr, enabled=True)
    assert decision["passed"] is False, f"{game}'s origin engine must be rejected"
    assert decision["reason"] in (
        "degenerate_engine_no_correct_changed_cells",
        "change_fidelity_below_threshold",
        "no_changing_transitions",
    )


def test_both_legacy_thresholds_are_reported_and_can_disagree():
    """The threshold ambiguity that moved a headline by ~10x must be unrepresentable.

    0.5 is the threshold recorded in the gap entry; 1.0 is what the agent actually ships
    (arc_competition_agent.py:5593, :5719). An artifact that reports only one of them
    cannot be checked, so both are emitted with the threshold named in the key.
    """
    rows, mask = _corpus_with_hud(n_changing=2, n_noop=10)
    # Masked, so the identity engine gets credit for the many genuine no-ops -- the exact
    # no-op-heavy shape in which lp85's near-identity engine reads 0.725 and ft09's reads
    # a perfect 1.0. Unmasked, the ticking counter makes identity wrong everywhere and the
    # two thresholds trivially agree, which would test nothing.
    vr = e3.WorldModelVerifier(rows, hud_mask=mask, hud_mask_enabled=True).score(_identity)
    d = e3.change_gate_decision(vr, enabled=True)
    assert d["legacy_accuracy_threshold_documented"] == 0.5
    assert d["legacy_accuracy_live_threshold"] == 1.0
    assert "5593" in d["legacy_accuracy_live_threshold_source"]
    # The identity engine on a no-op-heavy corpus is the exact case where the two verdicts
    # differ -- which is why reporting one alone was unfalsifiable.
    assert d["legacy_accuracy_would_pass"] != d["legacy_accuracy_would_pass_at_live_threshold"]
