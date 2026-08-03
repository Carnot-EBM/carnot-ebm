"""Pins the ROOT CAUSE of the exp5766-round-0 "6.33x regression" against exp5764.

REQ-ARC-WMTE-5766-REPRO.

The two experiments record a field named `heldout_accuracy` that is computed by two
different functions over two different row sets:

  exp5764  WorldModelVerifier(list(window)).score(engine).accuracy  -> the WHOLE window
  exp5766  _score_accuracy(_split_prefix_heldout(window)[1], engine) -> the LAST 1/3

These tests do not re-run any LLM. They pin the four mechanical facts that make the
reported gap an artefact rather than a mechanism regression, using constructed inputs
whose expected values are computable by hand:

  1. `_split_prefix_heldout` really does cut the LAST third.
  2. The two named quantities genuinely diverge for one and the same engine.
  3. A held-out tail made entirely of level-up rows scores 0.0 for a PERFECT engine --
     an unfalsifiable zero, not a measurement.
  4. The CEGIS loop shows its round-0 induce only the FIRST 2/3 of the window, and
     `prefix_accuracy` grades exactly those rows -- so exp5766 round 0 is an
     out-of-sample number fit on less evidence, while exp5764's is in-sample.

Each assertion is written so that deleting the behaviour it pins turns the test red;
see the module docstring of
`scripts/experiments/outer_loop_arc_round0_regression_repro_20260803.py` for the
measured run these were distilled from.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_executable_world_model import Transition, WorldModelVerifier
from carnot.agentic.arc_world_model_trust_energy import (
    _score_accuracy,
    _split_prefix_heldout,
)


def _t(grid, next_grid, *, action=1, level_before=0, level_after=0):
    return Transition(
        grid=np.asarray(grid, dtype=np.int16),
        action=int(action),
        data=None,
        next_grid=np.asarray(next_grid, dtype=np.int16),
        level_before=int(level_before),
        level_after=int(level_after),
    )


def _grid(v):
    return [[int(v), 0], [0, 0]]


def _identity(grid, action, data=None):
    return grid


def test_split_prefix_heldout_takes_the_last_third():
    """SCENARIO: the exp5766 metric is scored on the LAST 1/3 of the window.

    Pins the split itself. If `_split_prefix_heldout` ever stopped cutting the tail,
    the whole root-cause account below would no longer apply.
    """
    rows = [_t(_grid(i), _grid(i + 1), action=i) for i in range(9)]
    prefix, heldout = _split_prefix_heldout(rows)

    assert len(prefix) == 6, "prefix must be the first two thirds"
    assert len(heldout) == 3, "heldout must be the last third"
    # It is the TAIL specifically, not an arbitrary 3 rows.
    assert [t.action for t in heldout] == [6, 7, 8]
    assert prefix + heldout == rows, "prefix and heldout must PARTITION the window"


def test_whole_window_and_heldout_tail_are_different_quantities():
    """SCENARIO: one engine, two metrics, two very different numbers.

    This is the defect in one assertion. The engine is exactly right on the first two
    thirds and exactly wrong on the last third, which is the shape a model fit to rows
    it was SHOWN actually produces. exp5764 reads the first number; exp5766 reads the
    second; the artifacts compare them to each other.
    """
    # 6 no-op rows (identity is correct) then 3 changing rows (identity is wrong).
    rows = [_t(_grid(1), _grid(1), action=i) for i in range(6)]
    rows += [_t(_grid(1), _grid(2), action=6 + i) for i in range(3)]
    _prefix, heldout = _split_prefix_heldout(rows)

    whole_window = float(WorldModelVerifier(list(rows)).score(_identity).accuracy)
    tail_only = float(_score_accuracy(heldout, _identity))

    assert whole_window == 6 / 9, "exp5764's metric scores all 9 rows: 6 correct"
    assert tail_only == 0.0, "exp5766's metric scores only the 3 tail rows: 0 correct"
    assert whole_window > tail_only, (
        "the two fields named `heldout_accuracy` must not be assumed interchangeable; "
        "here the same engine reads 0.667 under one and 0.0 under the other"
    )


def test_all_levelup_heldout_tail_is_unfalsifiable_even_for_a_perfect_engine():
    """SCENARIO: 4 of the 13 roster games have a held-out tail with ZERO gradeable rows.

    `WorldModelVerifier.score` excludes level-up rows and returns `n_correct/max(1, n)`.
    When every tail row is a level-up, `n == 0` and the metric returns 0.0 for EVERY
    engine. Citing that 0.0 as evidence about an engine is failure mode "a prior
    negative may be unfalsifiable, not negative".

    The oracle here is a genuinely PERFECT engine, and the test proves it is not vacuous
    by showing it scores 1.0 on an ordinary corpus before showing it scores 0.0 here.
    """
    table = {}

    def oracle(grid, action, data=None):
        got = table.get((np.asarray(grid).tobytes(), int(action)))
        return got.copy() if got is not None else grid

    # CONTROL FIRST: the oracle must be able to score 1.0, or a 0.0 from it proves nothing.
    ordinary = [_t(_grid(1), _grid(2), action=0), _t(_grid(3), _grid(4), action=1)]
    for t in ordinary:
        table[(np.asarray(t.grid).tobytes(), int(t.action))] = np.asarray(t.next_grid)
    assert float(WorldModelVerifier(ordinary).score(oracle).accuracy) == 1.0, (
        "control is vacuous: the 'perfect' oracle cannot even score 1.0 on a normal corpus"
    )

    # THE UNFALSIFIABLE CASE: every tail row is a level-up, so every row is excluded.
    levelups = [
        _t(_grid(5), _grid(6), action=2, level_before=0, level_after=1),
        _t(_grid(7), _grid(8), action=3, level_before=1, level_after=2),
    ]
    for t in levelups:
        table[(np.asarray(t.grid).tobytes(), int(t.action))] = np.asarray(t.next_grid)

    vr = WorldModelVerifier(levelups).score(oracle)
    assert vr.n == 0, "every row must be excluded as a level-up row"
    assert vr.n_levelup_rows_excluded == 2
    assert float(vr.accuracy) == 0.0, (
        "a PERFECT engine scores 0.0 here -- the zero is structural, not a measurement"
    )
    assert float(_score_accuracy(levelups, oracle)) == 0.0, (
        "the exp5766 scoring entrypoint reports the same unfalsifiable zero"
    )


def test_cegis_round0_is_shown_only_the_prefix_and_prefix_accuracy_grades_it():
    """SCENARIO: exp5766's round 0 sees a third less evidence than exp5764's single shot.

    `execute_bounded_llm_reinduction` sets `induction_evidence = _proposal_prefix(...)`
    whenever the caller passes no `proposal_transitions` -- and `run_cegis_cell` does not.
    exp5764's `_induce_no_fence` passes the WHOLE window. So the two arms are fit on
    different amounts of evidence AND graded on different rows, which is why one is an
    in-sample fit number and the other an out-of-sample generalization number.

    The second assertion is what makes the reconstruction in the reproduction artifact
    sound: `_proposal_prefix` and `_split_prefix_heldout` must cut at the SAME index, or
    `prefix_accuracy` would not grade exactly the rows the model was shown.
    """
    import inspect

    from carnot.agentic.arc_llm_reinduction import _proposal_prefix
    from carnot.experiment_5760_cegis_refinement_induction_ab import run_cegis_cell

    assert "proposal_transitions" not in inspect.getsource(run_cegis_cell), (
        "run_cegis_cell must NOT pass proposal_transitions -- if it ever does, round 0 "
        "stops being prefix-only and this whole account needs re-deriving"
    )

    for n in (3, 4, 5, 9, 12, 25):
        rows = list(range(n))
        shown = _proposal_prefix(rows)
        graded_prefix, graded_heldout = _split_prefix_heldout(rows)
        assert shown == graded_prefix, (
            f"n={n}: prefix_accuracy must grade exactly the rows shown to the model"
        )
        assert len(shown) < n, f"n={n}: the CEGIS proposer must be shown FEWER rows than exist"
        assert len(graded_heldout) == n - len(shown)
