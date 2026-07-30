"""REQ-ARC-WMTE-6044: the win-state positive example, and the goal gate's successor generator.

Regression tests for four defects measured on 2026-07-29 against ka59's change-fidelity-1.0000
world model (git 341f776c9). Each test reproduces the ACTUAL incident shape, not a synthetic
happy path, per CLAUDE.md "QA-Layer Authenticity Discipline" ("write the regression test for the
exact incident/counterexample that motivated the check").

THE INCIDENTS.

1. `_goal_satisfiability_check`'s probe clicked the first 32 non-background cells in raw ROW-MAJOR
   RASTER order. Raster order walks into the top border first, so on ka59 all 32 clicks landed on
   row 21 while both movable blocks sat at rows 30-32 -- zero clicks inside a block, and ka59's only
   selection mechanic requires a click strictly inside one. A concept-correct goal was rejected as
   `degenerate_goal_predicate` after 2641 states. A pre-filter searching a strictly weaker action
   set than the planner it guards can only produce false negatives.

2. Nothing tested `goal(root) == True`. `lambda g: True` -- the most degenerate predicate
   expressible -- passed at depth 0 and the planner behind it returned a 1-action plan.

3. `_transitions_block` emitted `win.next_grid` labelled "is_level_complete must return True here".
   That is false: the completing action re-lays out the playfield atomically with winning (3527 of
   4096 cells on ka59, vs an ordinary-step median of 18.5), so `next_grid` is the NEXT LEVEL'S
   OPENING BOARD. The naive off-by-one fix (`win.grid`) is also wrong -- it is one action short of
   completion, where a correct predicate must be False.

4. `score_goal_predicate_consistency` graded level-up rows on `t.next_grid` and expected True, so it
   scored a CORRECT predicate as WRONG on exactly the rows carrying the positive signal.
"""

from typing import Any

import numpy as np
import pytest

from carnot.agentic import arc_llm_reinduction as reinduction
from carnot.agentic.arc_executable_world_model import (
    Transition,
    _transitions_block,
    objects_block,
    score_goal_predicate_consistency,
)


# ---------------------------------------------------------------------------------------------
# A minimal "click must land INSIDE the object" world, which is the ka59 mechanic that raster-order
# probing cannot satisfy. The board is deliberately shaped like the real failure: a full border of
# non-background cells (which raster order reaches first) and a small interior object (which the
# planner's component centroids reach, and which is the only thing a click can usefully hit).
# ---------------------------------------------------------------------------------------------
_BG = 1
_BORDER = 15
_OBJ = 7
_WON = 9

# Geometry mirrors the real ka59 incident: a horizontal border ABOVE the interactive object, on a
# board wide enough that the border row alone exceeds the probe's 32-click budget. On ka59 (64x64)
# the border is row 21 and both movable blocks are at rows 30-32, so raster order spends all 32
# clicks on row 21 and never descends to a block. A narrow test board would let raster order reach
# the object by accident and the test would assert nothing.
_W = 40
_BORDER_ROW = 10
_OBJ_TOP, _OBJ_LEFT, _OBJ_SIZE = 20, 20, 3


def _board() -> np.ndarray:
    grid = np.full((_W, _W), _BG, dtype=np.int16)
    grid[_BORDER_ROW, :] = _BORDER
    grid[_OBJ_TOP : _OBJ_TOP + _OBJ_SIZE, _OBJ_LEFT : _OBJ_LEFT + _OBJ_SIZE] = _OBJ
    return grid


def _engine(grid: np.ndarray, action: int, data: Any) -> np.ndarray:
    """Only a click landing strictly INSIDE the interior object does anything."""
    out = np.array(grid, copy=True)
    if action != 6 or not data:
        return out
    x, y = int(data.get("x", -1)), int(data.get("y", -1))
    if _OBJ_TOP <= y < _OBJ_TOP + _OBJ_SIZE and _OBJ_LEFT <= x < _OBJ_LEFT + _OBJ_SIZE:
        out[_OBJ_TOP : _OBJ_TOP + _OBJ_SIZE, _OBJ_LEFT : _OBJ_LEFT + _OBJ_SIZE] = _WON
    return out


def _goal(grid: np.ndarray) -> bool:
    return bool(np.any(np.asarray(grid) == _WON))


def test_raster_probe_cannot_reach_the_goal_but_component_probe_can():
    """INCIDENT 1. The board is reachable-in-one-click, and the OLD generator still misses it.

    This is the load-bearing assertion: it shows the fix is necessary, not merely different. The
    retained `_raster_probe_candidates` fallback is exercised directly so the contrast is measured
    inside one test rather than asserted from prose.
    """
    board = _board()

    # The goal IS reachable in a single click -- on the object's own centroid.
    centroid_x = _OBJ_LEFT + _OBJ_SIZE // 2
    centroid_y = _OBJ_TOP + _OBJ_SIZE // 2
    assert _goal(_engine(board, 6, {"x": centroid_x, "y": centroid_y})) is True

    # The shipped gate (component-aware) must find it.
    result = reinduction._goal_satisfiability_check(engine=_engine, goal=_goal, start_grid=board)
    assert result["satisfiable"] is True, (
        "the component-aware probe must reach a goal that is one click from the root"
    )

    # Now prove the ORIGINAL raster generator could not have. Rebuild the same BFS with the raster
    # candidates, which is what shipped before this fix.
    raster_clicks = []
    from collections import Counter

    flat = [int(v) for v in board.flatten().tolist()]
    background = Counter(flat).most_common(1)[0][0]
    coords = np.argwhere(board != background)
    for r, c in coords[:32]:
        raster_clicks.append((int(c), int(r)))

    assert len(raster_clicks) == 32
    inside = [
        (x, y)
        for x, y in raster_clicks
        if _OBJ_TOP <= y < _OBJ_TOP + _OBJ_SIZE and _OBJ_LEFT <= x < _OBJ_LEFT + _OBJ_SIZE
    ]
    assert inside == [], (
        "the incident shape: raster order fills its 32-click budget on the border and never "
        f"reaches the interior object; got {inside}"
    )
    # And therefore no raster click can satisfy the goal from the root.
    assert not any(_goal(_engine(board, 6, {"x": x, "y": y})) for x, y in raster_clicks), (
        "raster clicks must all be inert on the mechanic, which is why the gate false-negatived"
    )


def test_probe_falls_back_to_raster_when_component_generator_unavailable(monkeypatch):
    """The fix must DEGRADE, never crash, if the planner's generator is unavailable."""
    import carnot.agentic.arc_executable_world_model as e3

    def _boom(_grid):
        raise RuntimeError("component generator unavailable")

    monkeypatch.setattr(e3, "_model_candidates", _boom)

    # A goal reachable by the 5 keyboard actions alone, so the fallback path can still succeed and
    # we are testing "did not crash" rather than "found nothing".
    def keyboard_engine(grid: np.ndarray, action: int, _data: Any) -> np.ndarray:
        out = np.array(grid, copy=True)
        if action == 1:
            out[1, 1] = _WON
        return out

    result = reinduction._goal_satisfiability_check(
        engine=keyboard_engine, goal=_goal, start_grid=_board()
    )
    assert result["satisfiable"] is True


def test_goal_true_at_root_is_rejected_not_satisfied():
    """INCIDENT 2. `lambda g: True` used to pass at depth 0 and yield a 1-action plan."""
    result = reinduction._goal_satisfiability_check(
        engine=_engine, goal=lambda _g: True, start_grid=_board()
    )
    assert result["satisfiable"] is False
    assert result["counterexample"]["kind"] == "goal_predicate_true_at_root"


def test_constant_false_goal_is_still_rejected():
    """Guard: the root-true fix must not weaken the check it was added beside."""
    result = reinduction._goal_satisfiability_check(
        engine=_engine, goal=lambda _g: False, start_grid=_board()
    )
    assert result["satisfiable"] is False
    assert result["counterexample"]["kind"] == "degenerate_goal_predicate"


# ---------------------------------------------------------------------------------------------
# The prompt blocks. `_relayout` models the measured ka59 shape: the completing action rewrites
# most of the board because it also lays out the next level.
# ---------------------------------------------------------------------------------------------


def _win_transition() -> Transition:
    before = _board()
    after = np.full_like(before, 3)  # a full re-layout: the NEXT level's opening board
    after[0, 0] = _BORDER
    return Transition(
        grid=before,
        action=4,
        data=None,
        next_grid=after,
        level_before=0,
        level_after=1,
    )


def test_transitions_block_does_not_assert_the_relayout_frame_is_a_win_state():
    """INCIDENT 3. The old text claimed `is_level_complete must return True here` about the
    post-relayout frame, which is the next level's opening board."""
    win = _win_transition()
    block = _transitions_block([win])

    assert "WIN TRANSITION" in block
    # The exact false claim must be gone.
    assert "is_level_complete must return True here" not in block
    # It must name the ACTION, which is the trustworthy labelled signal.
    assert f"ACTION{win.action}" in block
    # It must state the constraint as a joint engine+goal condition.
    assert "is_level_complete(engine(" in block
    # And it must warn against the exact wrong inference.
    assert "re-lays out the board" in block


def test_transitions_block_relabels_the_opening_board_exemplar_truthfully():
    """The `previous_level_complete_grid` branch is captured AFTER the counter incremented, so it
    is the current level's opening board, not a level-complete state."""
    ordinary = Transition(
        grid=_board(),
        action=1,
        data=None,
        next_grid=_board(),
        level_before=0,
        level_after=0,
    )
    block = _transitions_block([ordinary], previous_level_complete_grid=_board())
    assert "WIN STATE EXEMPLAR" not in block
    assert "BOARD AT THE START OF THE CURRENT LEVEL" in block
    assert "is_level_complete must return False here" in block


def test_objects_block_does_not_label_either_frame_a_win_state():
    """INCIDENT 3, object-space half: the same poison was taught as an object table."""
    block = objects_block([_win_transition()])
    # ASSERT, DO NOT SKIP. `objects_block` returns "" on any internal failure (it is written to
    # never break the induction path), so an empty result here would silently disable every
    # assertion below -- the object-space half of this incident would stop being tested and the
    # suite would still report green. A skip is an invisible failure; a failing assertion is a
    # visible one. If `blob_topology` is genuinely unavailable in some future environment, this
    # test must fail loudly and be fixed, not quietly stop checking.
    assert block, (
        "objects_block returned empty -- the object-space assertions below would not run. "
        "This must fail rather than skip: a silently-disabled check is how the win-state "
        "poison survived in object space in the first place."
    )
    assert "WIN STATE OBJECTS" not in block
    assert "BOARD BEFORE THE COMPLETING ACTION OBJECTS" in block


def test_goal_repair_declines_a_fallback_that_is_trivially_true_at_root():
    """THE LIVE CASE: at a level boundary the exemplar IS the root, byte-identically.

    An earlier revision of this test used an exemplar with FEWER filled cells than the root and
    asserted GOAL-REPAIR declines. That shape CANNOT OCCUR LIVE, so the test certified the wrong
    thing: `previous_level_complete_grid` is set only by
    `arc_competition_agent._begin_level_goal_episode`, from
    `to_logical(grid_of(latest), detect_cell(grid_of(latest)))`, and the caller then immediately
    computes `self.root_grid` the same way from the SAME `latest`. The two grids are identical.

    With `>=` against an identical grid the fallback was trivially TRUE AT THE ROOT, so GOAL-REPAIR
    returned None on every reachable call -- an operator-directed mechanism (2026-06-25) silently
    dead. The strict bound is False at the root, so the repair fires again WITHOUT the root-true
    rejection being weakened by one iota.
    """
    root = _board()
    # THE LIVE SHAPE: exemplar and root are the same grid.
    exemplar = np.array(root, copy=True)
    assert np.array_equal(exemplar, root), "the live boundary case is exemplar == root"

    # An engine that can add a non-BACKGROUND cell, so "strictly fuller than the root" is reachable.
    # Note it fills a `_BG` cell, not a zero cell: this board's background is 1.
    def filling_engine(grid: np.ndarray, action: int, _data: Any) -> np.ndarray:
        out = np.array(grid, copy=True)
        if action == 1:
            background_cells = np.argwhere(out == _BG)
            if background_cells.size:
                r, c = background_cells[0]
                out[r, c] = _OBJ
        return out

    repaired = reinduction._repair_degenerate_goal(
        engine=filling_engine,
        previous_level_complete_grid=exemplar,
        root_grid=root,
    )
    assert repaired is not None, (
        "GOAL-REPAIR must fire in the live boundary case; returning None here is the dead-repair "
        "regression the strict bound fixes"
    )
    assert repaired["source"] == "exemplar_strictly_fuller_than_level_root_fallback"
    assert repaired["satisfiability"]["satisfiable"] is True
    # AND the repaired goal must not be the degenerate thing the root-true rule exists to reject.
    assert repaired["predicate"](root) is False, (
        "a repaired goal true at the root yields a zero-action plan -- worse than no repair"
    )


def test_goal_repair_still_declines_when_the_fallback_is_genuinely_unreachable():
    """Guard against over-loosening: the repair must still return None for TRUE unreachability.

    The fix must not become "always return a fallback". An engine that cannot add a filled cell
    makes "strictly fuller than the root" unreachable, and the honest answer is to give up this
    round rather than hand the planner a target it can never hit.
    """
    root = _board()

    def inert_engine(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
        return np.array(grid, copy=True)

    repaired = reinduction._repair_degenerate_goal(
        engine=inert_engine,
        previous_level_complete_grid=np.array(root, copy=True),
        root_grid=root,
    )
    assert repaired is None, "an unreachable fallback is not a repair"


def test_strictly_fuller_predicate_is_false_on_its_own_reference():
    """The one-line property the whole GOAL-REPAIR fix rests on, asserted directly.

    `_nonzero_count_predicate` uses `>=` and is therefore TRUE on its own reference grid; the strict
    variant must be FALSE there. That single difference is what makes the repaired goal survive the
    root-true rejection at a level boundary, where reference and root are the same grid.
    """
    grid = _board()
    assert reinduction._nonzero_count_predicate(grid)(grid) is True
    assert reinduction._strictly_fuller_than_predicate(grid)(grid) is False

    # Filling one BACKGROUND cell must satisfy it. This board's background is `_BG = 1`, NOT 0 --
    # which is exactly why the predicate counts non-MODAL cells rather than non-zero ones. A
    # count_nonzero version would be unreachable here (every cell is already non-zero), so the
    # repair would decline on ka59-shaped boards, i.e. on the real case.
    fuller = np.array(grid, copy=True)
    background_cells = np.argwhere(fuller == _BG)
    assert background_cells.size, "fixture must have background cells to fill"
    fuller[background_cells[0][0], background_cells[0][1]] = _OBJ
    assert reinduction._strictly_fuller_than_predicate(grid)(fuller) is True


def test_strictly_fuller_predicate_counts_non_modal_not_non_zero():
    """NAMED REGRESSION GUARD for the background assumption.

    On a board whose background is not colour 0, `count_nonzero` counts EVERY cell, so
    "strictly more filled cells" is unreachable and GOAL-REPAIR declines on every real board of that
    shape. This asserts the predicate is measured against the modal colour, so the distinction
    cannot be silently reverted to `count_nonzero` later.
    """
    grid = np.full((8, 8), 3, dtype=np.int16)  # background is 3; nothing is zero
    grid[0, 0] = 7
    assert np.count_nonzero(grid) == grid.size, "fixture: a count_nonzero version cannot grow"

    predicate = reinduction._strictly_fuller_than_predicate(grid)
    assert predicate(grid) is False
    more = np.array(grid, copy=True)
    more[1, 1] = 7  # one more non-background cell
    assert predicate(more) is True


def test_goal_only_prompt_does_not_claim_the_opening_board_is_a_win_state():
    """INCIDENT 3, worst instance. `_goal_only_prompt` is a FOCUSED win-condition prompt -- the
    model spends its entire budget on it -- and its single emphasised fact used to be
    "The level is COMPLETE at this WIN STATE grid (is_level_complete must return True here)" about
    `previous_level_complete_grid`, which is the CURRENT level's opening board.

    It also contradicted `_goal_satisfiability_check`: a model obeying that instruction produces a
    predicate true at the level root, which the gate now rejects outright.
    """
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    prompt = LocalGGUFProposer._goal_only_prompt(
        object.__new__(LocalGGUFProposer), "ka59", _board()
    )
    assert "WIN STATE" not in prompt
    assert "must return True" not in prompt
    assert "board at the START of the current level" in prompt
    assert "is_level_complete must return False here" in prompt


# ---------------------------------------------------------------------------------------------
# The grading veto.
# ---------------------------------------------------------------------------------------------


def test_levelup_row_graded_on_engine_counterfactual_not_rendered_frame():
    """INCIDENT 4. A CORRECT predicate is False on the re-laid-out frame, so the old grading
    marked it wrong on the one row that carries the positive signal."""
    win = _win_transition()

    # A STATE-DEPENDENT engine: the winning action advances a counter, and only the LAST advance
    # completes the level. State dependence is essential -- an engine whose winning action always
    # wins would make any same-action corroborating row also satisfy the goal, i.e. the goal really
    # would be too loose and the veto would be right to fire.
    _CTR = (0, 0)  # a HUD-ish counter cell, outside the object

    def engine(grid: np.ndarray, action: int, _data: Any) -> np.ndarray:
        out = np.array(grid, copy=True)
        if action == 4:
            out[_CTR] = min(9, int(out[_CTR]) + 1)
            if int(out[_CTR]) >= 3:
                out[_OBJ_TOP, _OBJ_LEFT] = _WON
        return out

    def correct_goal(grid: np.ndarray) -> bool:
        return bool(np.asarray(grid)[_OBJ_TOP, _OBJ_LEFT] == _WON)

    # Build the pre-win board: two advances, not yet won. The counter cell is zeroed explicitly --
    # this board's background is `_BG = 1`, so leaving it at the background value would start the
    # counter at 1 and win one step early.
    base = _board()
    base[_CTR] = 0
    step1 = engine(base, 4, None)
    step2 = engine(step1, 4, None)
    assert correct_goal(step2) is False, "fixture: not won until the third advance"

    # The real level-up row: the third advance wins, and the RENDERED frame is the next level's
    # re-layout (the whole point -- the terminal configuration is never drawn).
    win = Transition(
        grid=step2,
        action=4,
        data=None,
        next_grid=win.next_grid,
        level_before=0,
        level_after=1,
    )
    assert correct_goal(win.grid) is False
    assert correct_goal(win.next_grid) is False
    assert correct_goal(engine(win.grid, win.action, None)) is True

    # The counterfactual is only trusted when the engine has independently earned it: a measured
    # held-out fidelity at/above the floor AND corroboration on THIS action from a real non-level-up
    # row. Supply both, so this test exercises the trusted path rather than the ungradeable one.
    corroborating = Transition(
        grid=base,
        action=4,
        data=None,
        next_grid=step1,  # a REAL observed effect of action 4 that does NOT win
        level_before=0,
        level_after=0,
    )
    assert correct_goal(corroborating.next_grid) is False
    rows = [corroborating, win]

    old = score_goal_predicate_consistency(correct_goal, rows)
    new = score_goal_predicate_consistency(
        correct_goal, rows, engine=engine, engine_change_fidelity=1.0
    )

    # The incident: the correct predicate is scored WRONG on the level-up row under the old grading.
    # The corroborating no-op row IS graded right (goal False, no level-up), so 1 of 2.
    assert old.n_correct == 1
    assert old.accuracy == pytest.approx(0.5)
    assert old.n_levelups_graded_on_engine_counterfactual == 0

    # The fix: the level-up row is graded on the counterfactual and the correct predicate scores 1.0.
    assert new.n_correct == 2
    assert new.accuracy == pytest.approx(1.0)
    assert new.n_levelups_graded_on_engine_counterfactual == 1
    assert new.n_levelups_ungradeable_low_engine_fidelity == 0
    # And the independence cost is DISCLOSED rather than hidden.
    assert new.counterfactual_grading_is_not_oracle_distinct is True
    assert new.engine_fidelity_used_for_counterfactual_decision == pytest.approx(1.0)


def test_counterfactual_grading_is_refused_when_engine_fidelity_is_below_the_floor():
    """ORACLE-DISTINCTNESS. Grading a level-up row on the engine's counterfactual makes the goal
    veto depend on the engine, and engine + goal come from the SAME proposer in the SAME call -- so
    a jointly-confabulated pair agrees with itself and clears a veto that otherwise catches it.

    Measured: an engine that invents colour 9 on the winning action, paired with a goal testing for
    colour 9 (a colour in NO observed frame), scores 0.5 graded on `next_grid` (veto FIRES) and 1.0
    graded on the counterfactual (veto PASSES). So the counterfactual is gated on a goal-blind,
    independently-measured fidelity number, and below the floor the row is UNGRADEABLE -- not
    quietly graded on the wrong frame, which is the failure the counterfactual exists to fix.
    """
    win = _win_transition()

    def inventing_engine(grid: np.ndarray, action: int, _data: Any) -> np.ndarray:
        out = np.array(grid, copy=True)
        if action == win.action:
            out[:] = 9
        return out

    def matching_goal(grid: np.ndarray) -> bool:
        return bool(np.any(np.asarray(grid) == 9))

    low = score_goal_predicate_consistency(
        matching_goal, [win], engine=inventing_engine, engine_change_fidelity=0.1
    )
    assert low.n_levelups_graded_on_engine_counterfactual == 0
    assert low.n_levelups_ungradeable_low_engine_fidelity == 1
    assert low.counterfactual_grading_is_not_oracle_distinct is False
    reasons = [m.get("ungradeable") for m in low.mismatches if m.get("ungradeable")]
    assert reasons == ["engine_fidelity_below_counterfactual_floor"]
    # The ungradeable row leaves the DENOMINATOR too, so it cannot depress accuracy toward the veto.
    assert low.n == 0


def test_no_gradeable_rows_reads_as_consistent_not_as_maximally_inconsistent():
    """THE DENOMINATOR TRAP. With every level-up row dropped as ungradeable, `n == 0`, and the
    obvious `n_correct / max(1, n)` yields 0.0 -- which trips any `accuracy < threshold` veto. The
    veto would then fire HARDEST exactly when it has NO evidence, re-creating the
    reject-correct-predicates failure through the denominator instead of the numerator.

    This was a real bug in the first version of this fix, caught by a pre-existing live-veto test.
    """
    win = _win_transition()

    def inventing_engine(grid: np.ndarray, action: int, _data: Any) -> np.ndarray:
        out = np.array(grid, copy=True)
        if action == win.action:
            out[:] = 9
        return out

    # Only a level-up row, and the action is uncorroborated -> everything is dropped.
    result = score_goal_predicate_consistency(
        lambda g: bool(np.any(np.asarray(g) == 9)),
        [win],
        engine=inventing_engine,
        engine_change_fidelity=1.0,
    )
    assert result.n == 0
    assert result.accuracy == pytest.approx(1.0), (
        "an empty gradeable window must read as vacuously consistent, not as accuracy 0.0"
    )
    # `n_real_levelups` still counts the row, so the inertness is visible rather than disguised.
    assert result.n_real_levelups == 1
    assert result.n_levelups_ungradeable_low_engine_fidelity == 1


def test_counterfactual_refused_when_engine_is_unverified_on_the_winning_action():
    """The fidelity floor alone is NOT sufficient, and this is the condition that actually blocks
    the confabulation.

    A jointly-wrong pair confabulates on the WINNING action specifically -- exactly the action whose
    true effect is unobservable from `next_grid`. So an engine can score well overall and still be
    unconstrained there. Corroboration is therefore required from a real NON-level-up row using the
    same action, graded against REALITY (which is what keeps this check oracle-distinct).
    """
    win = _win_transition()

    def inventing_engine(grid: np.ndarray, action: int, _data: Any) -> np.ndarray:
        out = np.array(grid, copy=True)
        if action == win.action:
            out[:] = 9
        return out

    # A real non-level-up row on the winning action that the engine gets DEMONSTRABLY WRONG.
    contradicting = Transition(
        grid=_board(),
        action=win.action,
        data=None,
        next_grid=_board(),  # reality: nothing happened. The engine claims all-9s.
        level_before=0,
        level_after=0,
    )
    result = score_goal_predicate_consistency(
        lambda g: bool(np.any(np.asarray(g) == 9)),
        [contradicting, win],
        engine=inventing_engine,
        engine_change_fidelity=1.0,  # high overall fidelity must NOT be enough on its own
    )
    assert result.n_levelups_graded_on_engine_counterfactual == 0
    assert result.n_levelups_ungradeable_low_engine_fidelity == 1
    reasons = [m.get("ungradeable") for m in result.mismatches if m.get("ungradeable")]
    assert reasons == ["engine_unverified_on_this_action"]


def test_grading_default_is_byte_identical_without_an_engine():
    """The veto is live; passing no engine must preserve historical behaviour exactly."""
    win = _win_transition()
    noop = Transition(
        grid=_board(),
        action=1,
        data=None,
        next_grid=_board(),
        level_before=0,
        level_after=0,
    )
    rows = [win, noop]
    baseline = score_goal_predicate_consistency(lambda g: bool(np.any(g == 3)), rows)
    assert baseline.n == 2
    assert baseline.n_real_levelups == 1
    assert baseline.n_real_noops == 1
    # `next_grid` of the win row is the all-3 re-layout -> claimed True on a real level-up (correct
    # by the OLD rule) and False on the no-op -> 2/2 under historical grading.
    assert baseline.n_correct == 2
    assert baseline.n_levelups_graded_on_engine_counterfactual == 0


def test_grading_treats_a_broken_engine_as_ungradeable_not_as_the_rendered_frame():
    """A broken engine must not be laundered into a goal-predicate verdict -- and must not be
    laundered into a `next_grid` grading either.

    An earlier revision fell back to the historical frame here. That is the exact grading that
    penalises a CORRECT predicate, so "the engine crashed" would have been silently converted into
    "your win condition is wrong". The row is ungradeable; say so.
    """
    win = _win_transition()

    def broken(_grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
        raise RuntimeError("engine boom")

    result = score_goal_predicate_consistency(
        lambda g: bool(np.any(g == 3)), [win], engine=broken, engine_change_fidelity=1.0
    )
    assert result.n_levelups_graded_on_engine_counterfactual == 0
    assert result.n_levelups_ungradeable_low_engine_fidelity == 1
    assert result.n == 0


def test_grading_treats_a_wrong_shape_counterfactual_as_ungradeable():
    """A shape-changing engine cannot have produced this level's terminal board, so its
    counterfactual is unusable -- and `next_grid` is still the wrong frame. Ungradeable, not a
    silent fall back to the frame that penalises correctness."""
    win = _win_transition()

    def reshaping(_grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
        return np.zeros((2, 2), dtype=np.int16)

    # Corroborate the action so the refusal is attributable to the SHAPE, not to verification.
    corroborating = Transition(
        grid=np.zeros((2, 2), dtype=np.int16),
        action=win.action,
        data=None,
        next_grid=np.zeros((2, 2), dtype=np.int16),
        level_before=0,
        level_after=0,
    )
    result = score_goal_predicate_consistency(
        lambda g: bool(np.any(g == 3)),
        [corroborating, win],
        engine=reshaping,
        engine_change_fidelity=1.0,
    )
    assert result.n_levelups_graded_on_engine_counterfactual == 0
    assert result.n_levelups_ungradeable_low_engine_fidelity == 1


# ---------------------------------------------------------------------------------------------
# The DYNAMICS verifier -- the earlier, stricter gate on the same wrong frame.
# ---------------------------------------------------------------------------------------------


def test_dynamics_verifier_excludes_levelup_rows_from_grading():
    """`WorldModelVerifier.score` graded level-up rows against the re-laid-out `next_grid` too, and
    it runs BEFORE the goal checks at a stricter threshold (`heldout_accuracy >= threshold`), so it
    could reject an honest engine before any goal fix could matter.

    Measured with ONE identical, perfectly-honest engine: accuracy 1.0000 on a window with no
    level-up, 0.6667 on the same window whose last row is a real level-up. The engine is the same;
    only the presence of an unpredictable re-layout row differs.
    """
    from carnot.agentic.arc_executable_world_model import WorldModelVerifier

    def counter_engine(grid: np.ndarray, action: int, _data: Any) -> np.ndarray:
        out = np.array(grid, copy=True)
        if action == 2:
            out[0, 0] = min(9, int(out[0, 0]) + 1)
        return out

    s0 = np.zeros((6, 6), dtype=np.int16)
    s1 = counter_engine(s0, 2, None)
    s2 = counter_engine(s1, 2, None)
    relayout = np.zeros((6, 6), dtype=np.int16)
    relayout[1, 1] = 5  # the next level's opening board

    def row(grid, nxt, lvl_after):
        return Transition(
            grid=grid, action=2, data=None, next_grid=nxt, level_before=0, level_after=lvl_after
        )

    without_levelup = [row(s0, s1, 0), row(s1, s2, 0), row(s2, counter_engine(s2, 2, None), 0)]
    with_levelup = [row(s0, s1, 0), row(s1, s2, 0), row(s2, relayout, 1)]

    clean = WorldModelVerifier(without_levelup).score(counter_engine)
    assert clean.accuracy == pytest.approx(1.0)
    assert clean.n_levelup_rows_excluded == 0

    fixed = WorldModelVerifier(with_levelup).score(counter_engine)
    assert fixed.n_levelup_rows_excluded == 1
    assert fixed.n == 2, "the excluded row must leave the denominator, not just the numerator"
    assert fixed.accuracy == pytest.approx(1.0), (
        "an honest engine must not be penalised for causing the level-up we wanted"
    )
    assert fixed.change_fidelity == pytest.approx(1.0)


def test_dynamics_verifier_still_rejects_a_dishonest_engine_on_a_levelup_window():
    """The exclusion must not become a free pass: a wrong engine is still caught on the rows that
    DO carry gradeable dynamics."""
    from carnot.agentic.arc_executable_world_model import WorldModelVerifier

    def liar(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
        out = np.array(grid, copy=True)
        out[0, 0] = 8
        return out

    s0 = np.zeros((6, 6), dtype=np.int16)
    s1 = np.array(s0, copy=True)
    s1[3, 3] = 1
    relayout = np.zeros((6, 6), dtype=np.int16)
    relayout[1, 1] = 5
    rows = [
        Transition(grid=s0, action=2, data=None, next_grid=s1, level_before=0, level_after=0),
        Transition(grid=s1, action=2, data=None, next_grid=relayout, level_before=0, level_after=1),
    ]
    result = WorldModelVerifier(rows).score(liar)
    assert result.n_levelup_rows_excluded == 1
    assert result.accuracy == pytest.approx(0.0), "a dishonest engine must still be rejected"


# ---------------------------------------------------------------------------------------------
# The gate/planner budget unit.
# ---------------------------------------------------------------------------------------------


def test_goal_gate_budget_counts_engine_calls_like_plan_in_model():
    """The pre-veto must not be able to outsearch the planner it guards at the same `max_nodes`.

    `plan_in_model` does `nodes += 1` immediately after the engine call, BEFORE its shape check and
    BEFORE its `seen` dedup, so it budgets RAW ENGINE CALLS. The gate used to increment only AFTER
    the dedup, so it budgeted UNIQUE GRIDS -- a ~11x gap on ka59 (12435 unique vs 137347 calls),
    i.e. the gate was ~11x MORE PERMISSIVE than the search it guards. A gate looser than its planner
    certifies goals the planner then fails on, which mislabels the counterexample handed to
    `refactor()`.
    """

    def expanding_engine(grid: np.ndarray, action: int, _data: Any) -> np.ndarray:
        out = np.array(grid, copy=True)
        idx = int(out.sum()) % out.size
        out.flat[idx] = (int(out.flat[idx]) + int(action)) % 9
        return out

    root = np.zeros((8, 8), dtype=np.int16)
    for cap in (20, 100):
        result = reinduction._goal_satisfiability_check(
            engine=expanding_engine,
            goal=lambda _g: False,
            start_grid=root,
            max_nodes=cap,
            max_depth=50,
        )
        assert result["budget_unit"] == "engine_calls_matching_plan_in_model"
        assert result["engine_calls"] <= cap, (
            f"the gate spent {result['engine_calls']} engine calls against a {cap} budget"
        )

    # THE LOAD-BEARING ASSERTION. `engine_calls <= cap` alone does NOT pin the counter's position:
    # if it were incremented after the dedup it would equal the unique-grid count, which is also
    # <= cap, and the test would pass while the ~11x inversion was fully restored. So use a
    # HEAVILY-COLLAPSING engine, where raw calls vastly exceed unique grids, and assert the counter
    # tracks CALLS.
    def collapsing_engine(grid: np.ndarray, action: int, _data: Any) -> np.ndarray:
        # Every action maps into a tiny state set, so almost every successor is a duplicate.
        out = np.array(grid, copy=True)
        out[0, 0] = int(action) % 3
        return out

    collapsed = reinduction._goal_satisfiability_check(
        engine=collapsing_engine,
        goal=lambda _g: False,
        start_grid=root,
        max_nodes=10_000,
        max_depth=20,
    )
    calls = collapsed["engine_calls"]
    unique = collapsed["reachable_grids_evaluated"]
    assert calls > unique, (
        f"engine_calls ({calls}) must exceed unique grids ({unique}) on a collapsing engine -- "
        "equality means the budget is still counting post-dedup unique grids, which is the ~11x "
        "more-permissive-than-the-planner inversion"
    )
    # The unique-grid diagnostic is still reported -- informative, it just no longer decides when
    # to stop.
    assert unique >= 1
