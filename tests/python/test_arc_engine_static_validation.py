"""REQ-ARC-WMTE-6052 -- static + dry-run validation of an induced engine, before the trust gate.

EVERY CHECK IS PINNED TO ITS OBSERVED FAILURE, not to a synthetic case invented to make the
check look good. The four failures come from the 2026-07-30 gate-rejection audit
(`docs/research-notes/arc-gate-rejection-audit-2026-07-30.md`) and the 2026-07-31 budget sweep:

  * `test_ft09_frozen_engine_*`      -- ft09's REAL live engine, read off disk. Not a fixture.
  * `test_tu93_shape_*`              -- the "falls off the end of engine()" shape both tu93
                                        replicates had.
  * `test_lp85_unbound_local_*`      -- lp85's `UnboundLocalError` from `is_level_complete`,
                                        reconstructed (see the test's own docstring for why a
                                        reconstruction is the honest option here).
  * `test_truncation_*`              -- ft09 round 2's `HIT n_predict=4096 OUTPUT LIMIT`.

AND EVERY CHECK IS PINNED AGAINST ITS FALSE-POSITIVE DIRECTION. A validator that rejects code
is only as good as its false-positive rate, so `test_no_false_positive_on_*` covers the
constructs `_falls_through` deliberately refuses to flag. The corpus-scale version of the same
question -- 439 real generated engines, 9 flagged, 9 execution-confirmed, 0 unconfirmed -- lives
in `results/arc_engine_validation_20260731/corpus_scan.json`, because that evidence needs the
real corpus and cannot be asserted from a fixture.

THE ONE THING THESE TESTS MUST NOT BE READ AS SAYING: that a clean report means a good engine.
`test_identity_engine_passes_every_check` exists to pin the opposite, because the Phase-1 sweep
found that the completions scoring best on every structural check were the identity function.
"""

from __future__ import annotations

import pathlib
import textwrap

import numpy as np
import pytest

from carnot.agentic.arc_engine_static_validation import (
    EngineDefect,
    dry_run_defects,
    engine_changes_anything,
    missing_return_defects,
    repair_prompt_block,
    truncation_defect,
    validate_engine_code,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
FT09_FROZEN = (
    REPO
    / "results/arc_induce_budget_20260731/upstream_ft09_cells/on__ft09__s1__world_model.py.frozen"
)


class _T:
    """A `Transition` double: the dry run only needs `.grid`, `.action`, `.data`, `.next_grid`."""

    def __init__(self, grid, action, data=None, next_grid=None):
        self.grid = grid
        self.action = action
        self.data = data
        self.next_grid = grid if next_grid is None else next_grid


def _grid(h=8, w=8):
    g = np.zeros((h, w), dtype=int)
    g[2, 3] = 4
    return g


def _kinds(defects):
    return sorted({d.kind for d in defects})


# ---------------------------------------------------------------------------
# ft09 -- the real engine, off disk
# ---------------------------------------------------------------------------


def test_ft09_frozen_artifact_is_present():
    """The evidence this file's headline case rests on must exist, or these tests are theatre.

    Asserted rather than `skipif`-guarded ON PURPOSE. A `skipif` here would turn this file's
    single most important case into an invisible failure the day the artifact moved -- the whole
    ft09 evidence chain would go quiet and the suite would still print green.
    """
    assert FT09_FROZEN.exists(), f"missing the ft09 live engine artifact at {FT09_FROZEN}"


def test_ft09_frozen_engine_is_flagged_missing_return():
    """ft09's LIVE engine: `action != 6` returns, the `action == 6` path ends in comments.

    This is the engine that produced the 2026-07-30 episode's ft09 result. The audit describes
    it as "1112 of 1144 lines pure comment ... `engine()` ends with no `return` on its final
    path, so it returns None. No-op by omission."
    """
    src = FT09_FROZEN.read_text()
    defects = missing_return_defects(src)
    assert "missing_return" in _kinds(defects)
    (d,) = [x for x in defects if x.kind == "missing_return"]
    assert d.repairable is True
    assert d.line is not None


def test_ft09_frozen_engine_really_does_return_none():
    """The AST claim, cross-checked by EXECUTION -- an unreproducible claim would be worthless."""
    src = FT09_FROZEN.read_text()
    ns: dict = {"np": np, "numpy": np}
    exec(compile(src, "ft09", "exec"), ns)  # noqa: S102
    grid = np.zeros((64, 64), dtype=int)
    assert ns["engine"](grid.copy(), 6, {"x": 46, "y": 38}) is None
    # ... and the OTHER path is fine, so this is a per-path defect, not a broken file.
    assert np.asarray(ns["engine"](grid.copy(), 1, None)).shape == (64, 64)


def test_ft09_frozen_engine_dry_run_reports_returned_none():
    """The dry run reaches the same conclusion from behaviour, naming the action."""
    src = FT09_FROZEN.read_text()
    trans = [_T(np.zeros((64, 64), dtype=int), 6, {"x": 46, "y": 38})]
    defects = dry_run_defects(src, trans)
    assert "engine_returned_none" in _kinds(defects)
    assert "action=6" in [d for d in defects if d.kind == "engine_returned_none"][0].detail


# ---------------------------------------------------------------------------
# tu93 -- "scan for the player and fall off the end of engine() with no return"
# ---------------------------------------------------------------------------


TU93_SHAPE = textwrap.dedent(
    """
    import numpy as np

    def engine(grid, action, data):
        out = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 4:
                    if action == 1:
                        out[r, c] = 0
                        out[r - 1, c] = 4
                    elif action == 2:
                        out[r, c] = 0
                        out[r + 1, c] = 4

    def is_level_complete(grid):
        return bool(np.any(grid != 0))
    """
)


def test_tu93_shape_is_flagged_missing_return():
    """The audit's tu93 description in code: it scans, it mutates `out`, it never returns it.

    Both tu93 replicates failed this way, at held-out 0/8 and cell recall 0.0 in both arms.
    """
    assert "missing_return" in _kinds(missing_return_defects(TU93_SHAPE))


def test_tu93_shape_dry_run_confirms_none():
    trans = [_T(_grid(), 1), _T(_grid(), 2)]
    assert "engine_returned_none" in _kinds(dry_run_defects(TU93_SHAPE, trans))


def test_tu93_degenerate_goal_is_NOT_reported_here():
    """tu93's `is_level_complete` returns True for any non-empty grid -- and that is NOT our call.

    Degeneracy is a quality judgement made by `_goal_satisfiability_check` with a counterfactual
    this module does not have. Reporting it here would quietly turn a defect scanner into a
    second, weaker gate. The dry run must therefore stay silent about it.
    """
    trans = [_T(_grid(), 1)]
    kinds = _kinds(dry_run_defects(TU93_SHAPE, trans))
    assert not any(k.startswith("goal_") for k in kinds), kinds


# ---------------------------------------------------------------------------
# lp85 -- UnboundLocalError out of the GOAL predicate, on the root grid
# ---------------------------------------------------------------------------


LP85_SHAPE = textwrap.dedent(
    """
    import numpy as np

    def engine(grid, action, data):
        return grid.copy()

    def is_level_complete(grid):
        # `cell` is bound only inside the loop; on a grid with no cell of value 9 the
        # `return` below reads it unbound. This is the lp85 defect: generated code, not a
        # harness miscall -- a wrong arity would raise TypeError instead.
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 9:
                    cell = (r, c)
        return cell[0] == 0
    """
)


def test_lp85_unbound_local_is_caught_by_the_goal_dry_run():
    """lp85's observed failure, reconstructed -- and reconstructed deliberately.

    WHY A RECONSTRUCTION. The episode's engine store was a per-cell temp directory, so unlike
    ft09 (whose engine was frozen into the Phase-1 capture) lp85's actual source no longer
    exists. Rather than claim a preserved artifact this file does not have, the test encodes the
    exact mechanism the audit established: `UnboundLocalError: cannot access local variable
    'cell' where it is not associated with a value`, raised out of `is_level_complete`, on the
    ROOT grid (`reachable_grids_evaluated: 1` pins it there).

    The load-bearing part is WHERE it is caught. An engine-only dry run reports nothing for this
    file -- `engine` is a clean identity -- so the check that catches lp85 is the goal-predicate
    arm, and this test fails if that arm is removed.
    """
    trans = [_T(np.zeros((8, 8), dtype=int), 1)]
    defects = dry_run_defects(LP85_SHAPE, trans)
    kinds = _kinds(defects)
    assert "goal_raised" in kinds, kinds
    (d,) = [x for x in defects if x.kind == "goal_raised"]
    assert "UnboundLocalError" in d.detail
    assert d.repairable is True
    # The variable name survives into the repair prompt: that is what makes it a fix, not a veto.
    assert "cell" in d.detail


def test_lp85_shape_engine_arm_alone_would_miss_it():
    """Pins the near-miss: `engine` is clean here, so an engine-only scan reports nothing."""
    assert missing_return_defects(LP85_SHAPE) == []


def test_goal_returning_an_array_is_flagged_not_boolean():
    """A predicate returning a whole array is unusable as a truth value; `bool()` raises."""
    code = textwrap.dedent(
        """
        import numpy as np
        def engine(grid, action, data):
            return grid.copy()
        def is_level_complete(grid):
            return grid == 0
        """
    )
    assert "goal_not_boolean" in _kinds(dry_run_defects(code, [_T(_grid(), 1)]))


# ---------------------------------------------------------------------------
# truncation -- a missing observation, not a bad model
# ---------------------------------------------------------------------------


def test_truncation_flagged_when_capped_before_required_symbols():
    """ft09 round 2: `missing ('engine','is_level_complete') ... HIT n_predict=4096 OUTPUT LIMIT`."""
    d = truncation_defect(
        stop_type="limit",
        code="# a wall of comments and no functions\n",
        required=("engine", "is_level_complete"),
        budget=4096,
    )
    assert d is not None
    assert d.kind == "truncated_before_required_symbols"
    assert d.retryable is True
    assert d.repairable is False  # telling the model it ran out of room just spends more room
    assert "4096" in d.detail


def test_truncation_not_flagged_when_capped_but_complete():
    """Hitting the cap AFTER writing both functions loses nothing we needed."""
    code = "def engine(grid, action, data):\n    return grid\ndef is_level_complete(grid):\n    return False\n"
    assert (
        truncation_defect(
            stop_type="limit", code=code, required=("engine", "is_level_complete"), budget=4096
        )
        is None
    )


def test_truncation_not_flagged_on_a_natural_stop():
    assert truncation_defect(stop_type="word", code="", required=("engine",), budget=4096) is None


def test_truncation_short_circuits_the_other_checks():
    """A truncated file 'has no return' because it has no END. Saying so would be wrong.

    `validate_engine_code` must report the truncation ALONE, so the caller retries with more
    room instead of feeding a half-written file back as a repair.
    """
    half = "def engine(grid, action, data):\n    out = grid.copy()\n    # and then it was cut"
    defects = validate_engine_code(
        half, stop_type="limit", required=("engine", "is_level_complete"), budget=4096
    )
    assert _kinds(defects) == ["truncated_before_required_symbols"]


# ---------------------------------------------------------------------------
# FALSE-POSITIVE DIRECTION -- the constructs the checker must refuse to flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        # if/else where both arms return
        "    if action == 6:\n        return grid\n    else:\n        return grid.copy()",
        # a while True with no break never exits normally
        "    while True:\n        return grid",
        # try/finally: exit paths are runtime-dependent; do not flag
        "    try:\n        return grid\n    finally:\n        pass",
        # try/except where both arms return
        "    try:\n        return grid\n    except Exception:\n        return grid.copy()",
        # with-block whose body returns
        "    import contextlib\n    with contextlib.suppress(Exception):\n        return grid",
        # a raise is a legitimate terminal
        "    raise ValueError('unmodelled action')",
        # early return then unconditional return
        "    if action == 1:\n        return grid\n    return grid.copy()",
        # match with a wildcard where every case returns
        "    match action:\n        case 1:\n            return grid\n        case _:\n            return grid.copy()",
    ],
)
def test_no_false_positive_on_terminating_shapes(body):
    """Every one of these returns on all paths. Flagging any of them would REJECT working code.

    That is strictly worse than the status quo: a non-returning engine is already caught
    downstream (as a wrong prediction, with a misleading reason), whereas a false positive
    throws away an engine that would have worked.
    """
    code = f"import numpy as np\n\ndef engine(grid, action, data):\n{body}\n"
    assert missing_return_defects(code) == [], code


def test_loop_that_may_run_zero_times_IS_flagged():
    """The tu93 shape in miniature: a `for` can execute zero times, so it does not terminate."""
    code = (
        "def engine(grid, action, data):\n    for r in range(grid.shape[0]):\n        return grid\n"
    )
    assert "missing_return" in _kinds(missing_return_defects(code))


def test_while_true_with_a_break_IS_flagged():
    """A `while True` that can `break` falls out of the loop and off the end of the function."""
    code = (
        "def engine(grid, action, data):\n"
        "    while True:\n"
        "        if action == 1:\n"
        "            break\n"
    )
    assert "missing_return" in _kinds(missing_return_defects(code))


def test_try_except_falling_through_IS_flagged():
    """A try/except whose body AND every handler fall through plainly falls through.

    This is the one `try` shape the checker is confident about; the rest (anything with a
    `finally`) it deliberately declines to judge.
    """
    code = (
        "def engine(grid, action, data):\n"
        "    try:\n"
        "        x = grid[0, 0]\n"
        "    except Exception:\n"
        "        x = 0\n"
    )
    assert "missing_return" in _kinds(missing_return_defects(code))


def test_break_in_a_nested_loop_does_not_count_for_the_outer_one():
    """`_has_break` must bind a `break` to its own loop, or the outer `while True` looks exitable.

    Without the nesting rule this returns a false positive: the inner `break` would be read as
    letting the outer `while True` fall through, and a correct engine would be rejected.
    """
    code = (
        "def engine(grid, action, data):\n"
        "    while True:\n"
        "        for r in range(3):\n"
        "            break\n"
        "        return grid\n"
    )
    assert missing_return_defects(code) == []


def test_last_definition_wins():
    """Generated files really do define `engine` twice; Python binds the LAST one.

    Grading the first definition would grade a function that never runs -- in either direction.
    """
    bad_then_good = (
        "def engine(grid, action, data):\n    pass\n\n"
        "def engine(grid, action, data):\n    return grid\n"
    )
    good_then_bad = (
        "def engine(grid, action, data):\n    return grid\n\n"
        "def engine(grid, action, data):\n    x = 1\n"
    )
    assert missing_return_defects(bad_then_good) == []
    assert "missing_return" in _kinds(missing_return_defects(good_then_bad))


# ---------------------------------------------------------------------------
# the return-None literal, missing symbols, syntax
# ---------------------------------------------------------------------------


def test_a_nested_engine_does_not_shadow_the_real_one():
    """`exec` binds only TOP-LEVEL definitions, so a helper's inner `engine` is not the callable.

    This is not hypothetical bookkeeping: `ast.walk` is BREADTH-FIRST, so every module-level
    definition is visited before any nested one. A "last definition wins" rule written over
    `ast.walk` therefore picks the NESTED one whenever a nested one exists -- and a helper's
    throwaway inner function would decide the verdict for the real engine. Here the top-level
    engine is fine and the nested one is not; a walk-based rule reports a defect that the
    caller can never hit.
    """
    code = (
        "def make():\n"
        "    def engine(grid, action, data):\n"
        "        pass\n"
        "    return engine\n"
        "\n"
        "def engine(grid, action, data):\n"
        "    return grid\n"
    )
    assert missing_return_defects(code) == []


def test_a_nested_engine_is_used_when_there_is_no_top_level_one():
    """With no module-level `engine`, reporting on the nested body beats reporting nothing."""
    code = "def make():\n    def engine(grid, action, data):\n        pass\n    return engine\n"
    assert "missing_return" in _kinds(missing_return_defects(code))


def test_explicit_return_none_is_flagged():
    code = "def engine(grid, action, data):\n    if action == 6:\n        return None\n    return grid\n"
    assert "returns_none_literal" in _kinds(missing_return_defects(code))


def test_bare_return_is_flagged():
    code = "def engine(grid, action, data):\n    if action == 6:\n        return\n    return grid\n"
    assert "returns_none_literal" in _kinds(missing_return_defects(code))


def test_missing_engine_function_is_its_own_kind():
    assert _kinds(missing_return_defects("def is_level_complete(g):\n    return False\n")) == [
        "missing_function"
    ]


def test_syntax_error_is_reported_not_raised():
    defects = missing_return_defects("def engine(grid, action data):\n    return grid\n")
    assert _kinds(defects) == ["syntax_error"]
    assert defects[0].line is not None


def test_module_exec_failure_is_its_own_kind():
    """A NameError at module scope means the engine never existed -- a different repair."""
    code = "undefined_helper()\n\ndef engine(grid, action, data):\n    return grid\n"
    assert _kinds(dry_run_defects(code, [_T(_grid(), 1)])) == ["module_exec_raised"]


def test_wrong_shape_return_is_flagged():
    code = "import numpy as np\ndef engine(grid, action, data):\n    return np.zeros((3, 3), dtype=int)\n"
    defects = dry_run_defects(code, [_T(_grid(), 1)])
    assert "engine_wrong_shape" in _kinds(defects)


def test_dry_run_does_NOT_report_a_merely_wrong_prediction():
    """Being wrong about the game is the TRUST GATE's call, and duplicating it here would make
    this module a second gate with a worse metric. A well-formed but wrong engine is clean."""
    code = "import numpy as np\ndef engine(grid, action, data):\n    return grid * 0\n"
    trans = [_T(_grid(), 1, next_grid=_grid() + 7)]
    assert dry_run_defects(code, trans) == []


# ---------------------------------------------------------------------------
# the anti-claim: a clean report is NOT a quality claim
# ---------------------------------------------------------------------------


def test_identity_engine_passes_every_check():
    """The Phase-1 sweep's best-scoring completion was `return grid` on both branches.

    It is accepted, it parses, it returns on every path, it raises on nothing, and it models
    nothing. This test exists so that no future reader mistakes an empty defect list for a
    quality verdict.
    """
    code = (
        "def engine(grid, action, data):\n"
        "    if action == 6:\n        return grid\n    return grid\n"
        "def is_level_complete(grid):\n    return False\n"
    )
    trans = [_T(_grid(), 6, {"x": 1, "y": 1}), _T(_grid(), 1)]
    assert validate_engine_code(code, transitions=trans) == []
    # ... and the one honest signal about it is recorded separately, outside the defect list.
    assert engine_changes_anything(code, trans) is False


def test_engine_changes_anything_is_true_for_a_real_mechanic():
    code = (
        "import numpy as np\n"
        "def engine(grid, action, data):\n"
        "    out = grid.copy()\n"
        "    if action == 6:\n        out[0, 0] = 9\n"
        "    return out\n"
    )
    assert engine_changes_anything(code, [_T(_grid(), 6, {"x": 0, "y": 0})]) is True


def test_engine_changes_anything_is_none_when_unrunnable():
    assert engine_changes_anything("def engine(:\n", [_T(_grid(), 1)]) is None


# ---------------------------------------------------------------------------
# the repair prompt
# ---------------------------------------------------------------------------


def test_repair_prompt_carries_the_exception_text_and_omits_truncation():
    """The exception text is the whole value: it is what turns a veto into a fix.

    A truncation must NOT appear -- it is not repairable by telling the model about it, and
    describing it would consume the very budget the retry needs.
    """
    defects = [
        EngineDefect(
            kind="goal_raised",
            detail="`is_level_complete(grid)` raised UnboundLocalError: cannot access local variable 'cell'",
            repairable=True,
        ),
        EngineDefect(
            kind="truncated_before_required_symbols",
            detail="hit the 4096-token cap",
            retryable=True,
        ),
    ]
    block = repair_prompt_block(defects)
    assert "UnboundLocalError" in block
    assert "cell" in block
    assert "4096" not in block
    assert "same shape" in block.lower()


def test_repair_prompt_caps_the_echoed_code():
    """A repetition-loop runaway must NOT be echoed back in full.

    ft09's live engine is 1112 of 1144 lines of duplicated comment, and Phase 1 measured that a
    doubled budget leaves the DISTINCT emitted lines unchanged while the length doubles. Feeding
    the wall back would spend thousands of prompt tokens re-showing the model the exact text it
    is stuck repeating. The head is kept, the omission is stated so the model is not told a
    partial file is the whole file.
    """
    wall = "def engine(grid, action, data):\n" + ("    # the same comment\n" * 4000)
    block = repair_prompt_block(
        [EngineDefect(kind="missing_return", detail="d", repairable=True)],
        code=wall,
        max_code_chars=500,
    )
    assert len(block) < 2000, len(block)
    assert "further characters omitted" in block
    assert "def engine" in block  # the head, which carries the structure, survives


def test_repair_prompt_does_not_truncate_a_short_answer():
    short = "def engine(grid, action, data):\n    pass\n"
    block = repair_prompt_block(
        [EngineDefect(kind="missing_return", detail="d", repairable=True)],
        code=short,
        max_code_chars=4000,
    )
    assert "omitted" not in block
    assert short.strip() in block


def test_repair_prompt_is_empty_when_nothing_is_repairable():
    """No repairable defect must produce NO block -- an empty instruction appended to a prompt
    is worse than none, because it spends tokens telling the model nothing."""
    assert repair_prompt_block([EngineDefect(kind="x", detail="y", retryable=True)]) == ""
    assert repair_prompt_block([]) == ""


def test_repair_prompt_can_include_the_failing_code():
    block = repair_prompt_block(
        [EngineDefect(kind="missing_return", detail="d", repairable=True)],
        code="def engine(g, a, d):\n    pass\n",
    )
    assert "```python" in block
    assert "def engine" in block


# ---------------------------------------------------------------------------
# validate_engine_code ordering
# ---------------------------------------------------------------------------


def test_validate_skips_the_dry_run_when_there_are_no_transitions():
    """The caller may not have transitions yet; the static arm must still run."""
    code = "def engine(grid, action, data):\n    x = 1\n"
    assert "missing_return" in _kinds(validate_engine_code(code, transitions=None))


def test_validate_returns_empty_for_well_formed_code():
    code = (
        "import numpy as np\n"
        "def engine(grid, action, data):\n    return grid.copy()\n"
        "def is_level_complete(grid):\n    return bool(np.all(grid == 0))\n"
    )
    assert validate_engine_code(code, transitions=[_T(_grid(), 1)]) == []
