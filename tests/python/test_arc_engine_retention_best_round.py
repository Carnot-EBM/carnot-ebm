"""REQ-ARC-WMTE-6035 / SCENARIO-ARC-WMTE-6035-RETAIN-BEST-ROUND.

WHAT THESE TESTS PROTECT, AND WHY THEY EXIST.

`execute_bounded_llm_reinduction` runs up to three attempts at a world model. Round 1 induces;
rounds 2..N call `proposer.refactor()`, and every one of those overwrites
`results/arc_e3/<game>/world_model.py` in place. Before REQ-ARC-WMTE-6035 the loop assumed the
LAST round is the BEST round, in two separate places:

  (i)  it left round N's source on disk, so the next `load_engine()` -- next stall, next level,
       next episode -- started from the last refactor however bad it was; and
  (ii) it reassigned `last_engine = selected.engine` unconditionally, so an in-memory caller
       that never touches the store was handed round N's engine too.

Refinement is not monotone. Replaying the real historical write sequence of six games' engines
put last-write-wins at mean change_fidelity 0.0042 (change gate 0 of 12) against retain-best at
0.3979 (5 of 12), with ka59's engine having peaked at 1.0000 and sitting at 0.0000 on disk. On
this very function, a scripted proposer replaying real recorded ka59 induction blobs produced
per-round held-out accuracies 0.65 -> 0.15 -> 0.075 and the loop returned round 3.

Each test below is written so that it FAILS if a specific part of the fix is removed. The
mutations they are proven against are named in each docstring.

EVIDENCE SAFETY: every test redirects the engine store to `tmp_path`. Nothing here can write
`results/arc_e3`, which is measurement evidence and must never be mutated by a test run.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_llm_reinduction as _reinduction
from carnot.agentic.arc_executable_world_model import Transition
from carnot.agentic.arc_llm_reinduction import (
    _retention_signal,
    engine_retention_enabled,
    execute_bounded_llm_reinduction,
)


@pytest.fixture(autouse=True)
def _multi_round_cap(monkeypatch):
    """Retention only has meaning across MULTIPLE rounds, so these tests need the pre-cap bound.

    MAX_REFINEMENT_ROUNDS was capped 3 -> 1 on 2026-08-17 (operator-approved: rounds past the
    first measured pooled-negative on held-out). At 1 the executor never reaches a second round,
    so "retain the BEST round" has nothing to choose between and every test here fails on a
    round-count assertion rather than on the behaviour it is checking.

    Raising the bound for this module keeps these tests measuring retention. It does not weaken
    the cap: the shipped default is pinned separately by
    tests/python/test_arc_refinement_rounds_cap.py, and the executor reads the module global at
    call time so patching it is sufficient.
    """
    monkeypatch.setattr(_reinduction, "MAX_REFINEMENT_ROUNDS", 3)


GAME = "retn"

# ---------------------------------------------------------------------------------------------
# A synthetic game with genuinely-changing dynamics, so the held-out split has changed cells to
# score on. A player (value 3) moves on a 6x6 torus; action 3 additionally ticks a counter cell
# at (5, 5). Every transition changes at least two cells, which is what makes
# `score_change_weighted_consistency` informative here.
# ---------------------------------------------------------------------------------------------

_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
# Fixed, not sampled: the prefix/held-out split is 6/3, and the held-out tail must contain an
# action-3 tick so the good engine's exact accuracy is BELOW the live 1.0 threshold. Without
# that the first round would pass the dynamics veto and the loop would never reach a round 2 --
# i.e. the regression this REQ fixes would be unreachable from the test.
ACTIONS = [1, 3, 0, 2, 1, 3, 0, 2, 3]


def _true_next(grid: np.ndarray, action: int) -> np.ndarray:
    g = grid.copy()
    pos = np.argwhere(g == 3)
    r, c = int(pos[0][0]), int(pos[0][1])
    dr, dc = _MOVES[int(action) % 4]
    g[r, c] = 0
    g[(r + dr) % g.shape[0], (c + dc) % g.shape[1]] = 3
    if int(action) % 4 == 3:
        g[5, 5] = int(g[5, 5]) + 1
    return g


def _corpus() -> tuple[list[Transition], np.ndarray]:
    grid = np.zeros((6, 6), dtype=int)
    grid[2, 2] = 3
    root = grid.copy()
    rows: list[Transition] = []
    for action in ACTIONS:
        nxt = _true_next(grid, action)
        rows.append(
            Transition(
                grid=grid.copy(),
                action=action,
                data=None,
                next_grid=nxt.copy(),
                level_before=0,
                level_after=0,
            )
        )
        grid = nxt
    return rows, root


# The GOOD engine: models the move exactly, does NOT model the action-3 counter tick. Measured
# on the corpus above: heldout_accuracy 0.6667 (fails the live 1.0 dynamics veto, so the loop
# refactors) and heldout_change_consistency 0.8571 (the retention signal).
GOOD_SRC = """
import numpy as np

_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def engine(grid, action, data):
    g = np.asarray(grid).copy()
    pos = np.argwhere(g == 3)
    if len(pos) == 0:
        return g
    r, c = int(pos[0][0]), int(pos[0][1])
    dr, dc = _MOVES.get(int(action) % 4, (0, 0))
    g[r, c] = 0
    g[(r + dr) % g.shape[0], (c + dc) % g.shape[1]] = 3
    return g


def is_level_complete(grid):
    g = np.asarray(grid)
    pos = np.argwhere(g == 3)
    return bool(len(pos) and int(pos[0][0]) == 0 and int(pos[0][1]) == 0)
"""

# A worse refactor: a no-op engine. heldout_change_consistency 0.0, and a constant-False goal.
WORSE_SRC = """
import numpy as np


def engine(grid, action, data):
    return np.asarray(grid).copy()


def is_level_complete(grid):
    return False
"""

# A different worse refactor, so round 2 and round 3 are distinguishable ON DISK by bytes.
WORST_SRC = """
import numpy as np


def engine(grid, action, data):
    g = np.asarray(grid).copy()
    g[0, 0] = 1
    return g


def is_level_complete(grid):
    return False
"""


# A CRASHING engine: models the move exactly, but RAISES on the action-3 tick transitions.
# `score_change_weighted_consistency` swallows the exception with a `continue`, so those
# transitions contribute to NEITHER `correct_changed_cells` NOR `true_changed_cells`. The result
# is a high consistency measured over a SMALLER denominator -- which is why that denominator is
# an engine-dependent diagnostic and not a pure property of the corpus.
CRASHY_SRC = """
import numpy as np

_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def engine(grid, action, data):
    if int(action) % 4 == 3:
        raise ValueError("this engine does not model the counter tick")
    g = np.asarray(grid).copy()
    pos = np.argwhere(g == 3)
    if len(pos) == 0:
        return g
    r, c = int(pos[0][0]), int(pos[0][1])
    dr, dc = _MOVES.get(int(action) % 4, (0, 0))
    g[r, c] = 0
    g[(r + dr) % g.shape[0], (c + dc) % g.shape[1]] = 3
    return g


def is_level_complete(grid):
    return False
"""


def _prefix_overfit_src(transitions: list[Transition]) -> str:
    """An engine that MEMORISES the prefix and no-ops everywhere else.

    Its prefix change-consistency is 1.0 and its held-out change-consistency is 0.0. It exists
    so a test can prove retention ranks rounds on the HELD-OUT split rather than on the prefix
    the proposer was shown or on the full corpus -- ranking on the prefix is exactly the
    "select on what the model already saw" mistake, and it would make this engine win.
    """

    prefix = transitions[:6]
    pairs = [
        (np.asarray(t.grid).tolist(), np.asarray(t.next_grid).tolist(), int(t.action))
        for t in prefix
    ]
    return f"""
import numpy as np

_MEMO = {pairs!r}


def engine(grid, action, data):
    g = np.asarray(grid)
    for before, after, act in _MEMO:
        if int(act) == int(action) and np.array_equal(g, np.asarray(before)):
            return np.asarray(after)
    return g.copy()


def is_level_complete(grid):
    return False
"""


class _ScriptedProposer:
    """Replays a fixed list of engine sources, one per round, writing each into the store.

    This is the shape of the real failure: `induce` writes round 1, each `refactor` overwrites
    it. Nothing about the retention fix may depend on the proposer being an LLM.
    """

    model_specs = "scripted-retention-test-proposer"

    def __init__(self, store: Path, game: str, sources: list[str]) -> None:
        self.store = Path(store)
        self.game = game
        self.sources = list(sources)
        self.writes: list[int] = []

    def _write(self, index: int) -> tuple[bool, str]:
        src = self.sources[min(index, len(self.sources) - 1)]
        path = self.store / self.game / "world_model.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(src)
        self.writes.append(index)
        return True, f"wrote round {index + 1}"

    def induce(self, game, trans, cell, *, previous_level_complete_grid=None):
        return self._write(0)

    def refactor(self, game, vr):
        return self._write(len(self.writes))


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sources: list[str],
    *,
    min_heldout_accuracy: float = 1.0,
    plan_in_model=None,
    corpus=None,
):
    """Drive the REAL `execute_bounded_llm_reinduction` against the scripted proposer.

    The selection signal comes from the real `select_trusted_world_model`; nothing about the
    scoring path is stubbed, because the whole claim under test is about which round's engine
    that real signal retains.
    """

    monkeypatch.setattr(e3, "E3_DIR", tmp_path)
    transitions, root = corpus if corpus is not None else _corpus()
    proposer = _ScriptedProposer(tmp_path, GAME, sources)
    result = execute_bounded_llm_reinduction(
        game=GAME,
        transitions=transitions,
        cell=1,
        root_grid=root,
        proposer=proposer,
        candidate_provider=lambda engine, goal: [("loaded_world_model.py", engine, goal)],
        load_engine=e3.load_engine,
        plan_in_model=plan_in_model or (lambda engine, goal, grid: None),
        max_rounds=3,
        min_heldout_accuracy=min_heldout_accuracy,
    )
    store_text = (tmp_path / GAME / "world_model.py").read_text()
    return result, store_text, proposer, root


def _predict(engine, grid: np.ndarray, action: int) -> np.ndarray:
    return np.asarray(engine(np.asarray(grid).copy(), action, None))


def _all_noop_corpus() -> tuple[list[Transition], np.ndarray]:
    """A corpus in which NOTHING ever changes -- the ft09-shaped degenerate held-out split.

    Measured on the real corpus (`hostile_noop_heldout_results.json`, 55cbd98a7baf3cd4): ft09
    seed 0 has 0 of 40 held-out transitions changing, so `true_changed_cells` is 0, the retention
    signal is 0.0 for EVERY engine including a perfect one, and a do-nothing engine nonetheless
    scores `heldout_accuracy` 1.0 and clears the shipped gate. This is that situation in
    miniature, so honest limit 1 is an executable fact rather than a comment.
    """

    grid = np.zeros((6, 6), dtype=int)
    grid[2, 2] = 3
    root = grid.copy()
    rows = [
        Transition(
            grid=grid.copy(),
            action=action,
            data=None,
            next_grid=grid.copy(),
            level_before=0,
            level_after=0,
        )
        for action in ACTIONS
    ]
    return rows, root


# =============================================================================================
# DEFECT (i): the STORE must end holding the retained round, not the last round.
# =============================================================================================


def test_store_retains_the_best_round_not_the_last_write(monkeypatch, tmp_path):
    """SCENARIO-ARC-WMTE-6035-RETAIN-BEST-ROUND (on disk).

    MUTATION PROOF: delete the `_retain_engine_source_on_disk` write (or make
    `_finalise_engine_retention` a no-op) and the store holds WORST_SRC -- exactly the
    last-write-wins state that put ka59 at change_fidelity 0.0000.
    """

    result, store_text, proposer, _root = _run(
        monkeypatch, tmp_path, [GOOD_SRC, WORSE_SRC, WORST_SRC]
    )

    assert proposer.writes == [0, 1, 2], "the loop must actually have run three rounds"
    assert result.refinement_rounds_used == 3
    assert store_text == GOOD_SRC, "the store must hold round 1, the highest-signal round"
    assert store_text != WORST_SRC
    assert result.engine_retention["restored"] is True
    assert result.engine_retention["best_round"] == 1
    assert result.engine_retention["enabled"] is True


def test_store_write_is_a_noop_when_the_last_round_is_already_the_best(monkeypatch, tmp_path):
    """Retention must not churn the store when there is nothing to roll back.

    MUTATION PROOF: make `_retain_engine_source_on_disk` write unconditionally and `restored`
    becomes True here, i.e. the fix would rewrite the store on every single call.
    """

    _result, store_text, _proposer, _root = _run(
        monkeypatch, tmp_path, [WORSE_SRC, WORST_SRC, GOOD_SRC]
    )

    assert store_text == GOOD_SRC
    assert _result.engine_retention["restored"] is False
    assert _result.engine_retention["reason"] == "store_already_holds_retained_engine"
    assert _result.engine_retention["best_round"] == 3


# =============================================================================================
# DEFECT (ii): the RETURNED engine must be the retained round, for an in-memory caller that
# never reads the store at all.
# =============================================================================================


def test_returned_engine_is_the_best_round_not_the_last_round(monkeypatch, tmp_path):
    """SCENARIO-ARC-WMTE-6035-RETAIN-BEST-ROUND (in memory).

    MUTATION PROOF: restore `last_engine = selected.engine` as an unconditional assignment and
    the returned engine is WORST's (which writes a 1 into the corner and never moves the
    player), so the movement assertion below fails.
    """

    result, _store_text, _proposer, root = _run(
        monkeypatch, tmp_path, [GOOD_SRC, WORSE_SRC, WORST_SRC]
    )

    assert result.planned is False
    assert result.engine is not None
    predicted = _predict(result.engine, root, 1)
    assert np.array_equal(predicted, _true_next(root, 1)), (
        "the returned engine must be round 1's mover, not round 3's corner-writer"
    )
    # And the scalar diagnostics must describe the SAME round, not a discarded one.
    assert result.heldout_accuracy == pytest.approx(0.6667, abs=1e-3)
    assert result.selected_candidate_name == "loaded_world_model.py"


def test_goal_diagnostics_describe_the_retained_round(monkeypatch, tmp_path):
    """The reported goal must belong to the returned engine.

    Run with the dynamics veto disabled so every round reaches the goal-satisfiability check:
    round 1's goal (player reaches the origin) IS satisfiable, rounds 2 and 3 are constant-False.

    MUTATION PROOF: drop the `if is_best:` guard on `last_goal_satisfiable` /
    `last_goal_satisfiability` and the result reports round 3's unsatisfiable goal next to
    round 1's engine -- a self-contradictory artifact, and the live agent installs a goal bias
    off exactly this field.
    """

    result, _store_text, _proposer, root = _run(
        monkeypatch,
        tmp_path,
        [GOOD_SRC, WORSE_SRC, WORST_SRC],
        min_heldout_accuracy=0.0,
    )

    assert result.refinement_rounds_used == 3
    assert result.goal_predicate_satisfiable is True
    assert result.goal_predicate is not None
    assert result.goal_satisfiability.get("satisfiable") is True
    # The retained engine and the retained goal are consistent: the goal is about the player
    # reaching the origin, and only round 1's engine moves the player at all.
    assert np.array_equal(_predict(result.engine, root, 1), _true_next(root, 1))


# A mover whose goal predicate is constant-False, so the round it comes from can never plan.
# Its dynamics are GOOD's, so it wins the retention comparison outright.
GOOD_ENGINE_DEGENERATE_GOAL_SRC = GOOD_SRC.replace(
    """def is_level_complete(grid):
    g = np.asarray(grid)
    pos = np.argwhere(g == 3)
    return bool(len(pos) and int(pos[0][0]) == 0 and int(pos[0][1]) == 0)""",
    """def is_level_complete(grid):
    return False""",
)

# A WEAKER mover: correct for up/down/left, wrong for right (it moves down instead). Lower
# held-out change-consistency than GOOD, but still able to walk to the origin, so the round it
# comes from CAN plan while a higher-signal earlier round cannot.
WEAK_MOVER_SATISFIABLE_GOAL_SRC = """
import numpy as np

_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (1, 0)}


def engine(grid, action, data):
    g = np.asarray(grid).copy()
    pos = np.argwhere(g == 3)
    if len(pos) == 0:
        return g
    r, c = int(pos[0][0]), int(pos[0][1])
    dr, dc = _MOVES.get(int(action) % 4, (0, 0))
    g[r, c] = 0
    g[(r + dr) % g.shape[0], (c + dc) % g.shape[1]] = 3
    return g


def is_level_complete(grid):
    g = np.asarray(grid)
    pos = np.argwhere(g == 3)
    return bool(len(pos) and int(pos[0][0]) == 0 and int(pos[0][1]) == 0)
"""


def test_a_planned_return_reports_its_own_round_not_the_retained_one(monkeypatch, tmp_path):
    """A PLANNED return hands back THIS round's engine, goal and plan -- so its diagnostics
    must be this round's too.

    Round 1 has the better dynamics (it IS the retained round) but a constant-False goal, so it
    cannot plan. Round 2 is a weaker mover with a reachable goal, and it plans. The result must
    report round 2's `goal_predicate_satisfiable=True` next to round 2's `goal_predicate` --
    reporting round 1's `False` there would contradict the very plan being returned, and the
    live agent gates `_install_goal_bias` on exactly that field.

    MUTATION PROOF: bind the planned-return `goal_predicate_satisfiable` /
    `goal_satisfiability` back to `last_goal_satisfiable` / `last_goal_satisfiability` and this
    fails. (This is not hypothetical: the first draft of REQ-ARC-WMTE-6035 did exactly that and
    was caught by
    `test_experiment_4664_l2_goal_predicate_induction_live.py::
    test_scenario_arc_wmte_4664_degenerate_goal_rejected_before_planning`.)
    """

    def plan_in_model(engine, goal, grid):
        # Up, up, left, left: (2,2) -> (0,0), using only the moves the weak mover gets right.
        return [
            {"action": 0, "data": None},
            {"action": 0, "data": None},
            {"action": 2, "data": None},
            {"action": 2, "data": None},
        ]

    result, _store_text, _proposer, root = _run(
        monkeypatch,
        tmp_path,
        [GOOD_ENGINE_DEGENERATE_GOAL_SRC, WEAK_MOVER_SATISFIABLE_GOAL_SRC, WORST_SRC],
        min_heldout_accuracy=0.0,
        plan_in_model=plan_in_model,
    )

    assert result.planned is True
    assert result.refinement_rounds_used == 2
    # The retained engine is round 1's -- retention did NOT follow the planning round.
    assert result.engine_retention["best_round"] == 1
    # ...and yet the reported goal diagnostics belong to round 2, the round that planned.
    assert result.goal_predicate_satisfiable is True
    assert result.goal_satisfiability.get("satisfiable") is True
    at_origin = np.zeros_like(root)
    at_origin[0, 0] = 3
    assert result.goal_predicate is not None
    assert bool(result.goal_predicate(at_origin)) is True, (
        "the returned goal must be round 2's reachable one, not round 1's constant-False stub"
    )
    assert bool(result.goal_predicate(root)) is False
    # The returned engine is the planning round's weak mover: it gets "right" wrong.
    assert not np.array_equal(_predict(result.engine, root, 3), _true_next(root, 3))


# =============================================================================================
# THE SELECTION SIGNAL: held-out, runtime-visible, and nothing else.
# =============================================================================================


def test_retention_signal_reads_heldout_change_consistency_only():
    """`_retention_signal` must not fall back to some other score.

    MUTATION PROOF: point it at `heldout_accuracy` (or at `prefix_change_consistency`) and this
    returns 0.99 (or 1.0) instead of 0.25. Those are the two quantities a well-meaning edit
    would reach for, and both would make the shipped selection rule differ from the measured
    one.
    """

    selection = SimpleNamespace(
        selected_score=SimpleNamespace(
            heldout_change_consistency=0.25,
            heldout_accuracy=0.99,
            prefix_change_consistency=1.0,
            trust_energy=3.0,
        )
    )
    assert _retention_signal(selection) == pytest.approx(0.25)


def test_retention_signal_does_not_fall_back_when_the_field_is_zero_or_absent():
    """The docstring's no-fallback guarantee, tested at the two inputs that actually exercise it.

    The case above pins "reads the right field" but never runs the fallback branch, because
    0.25 is truthy. The guarantee that keeps retention NON-CIRCULAR is the other half: when
    `heldout_change_consistency` is uninformative, the answer is 0.0 -- every round then ties
    and the tie rule keeps the incumbent -- NOT a quiet substitution of some other metric.

    Both inputs here are the live regime, not a contrivance:

      * ZERO. `consistency` is correct_changed_cells / max(1, true_changed_cells), so a
        held-out split with no changing transition scores 0.0 for every engine, INCLUDING a
        perfect one (ft09 seed 0: 0 of 40 held-out transitions change). The signal is 0.0 for
        every round in 23 of the counterfactual's 55 windows -- 42% -- which is exactly where
        the tie rule governs the outcome.
      * ABSENT. An injected or older score object need not carry the field at all.

    MUTATION PROOF (both die on this test, neither dies on the one above):
      HM3  `... heldout_change_consistency, 0.0) or getattr(score, "heldout_accuracy", 0.0)
            or 0.0` -- the shipped line already ends in `or 0.0`, so one more `or` reads as a
            natural edit, and it fires on a legitimate 0.0. Returns 0.99 here.
      HM6  a `if not hasattr(score, "heldout_change_consistency")` guard returning
            `heldout_accuracy`. Returns 0.99 on the absent case.
    """

    zero_beside_a_high_accuracy = SimpleNamespace(
        selected_score=SimpleNamespace(
            heldout_change_consistency=0.0,
            heldout_accuracy=0.99,
            prefix_change_consistency=1.0,
            trust_energy=3.0,
        )
    )
    assert _retention_signal(zero_beside_a_high_accuracy) == pytest.approx(0.0)

    field_absent = SimpleNamespace(
        selected_score=SimpleNamespace(
            heldout_accuracy=0.99,
            prefix_change_consistency=1.0,
            trust_energy=3.0,
        )
    )
    assert not hasattr(field_absent.selected_score, "heldout_change_consistency")
    assert _retention_signal(field_absent) == pytest.approx(0.0)


def test_an_uninformative_zero_signal_is_distinguishable_from_a_bad_one(monkeypatch, tmp_path):
    """Honest limit 1, made checkable in the artifact instead of only in a comment.

    A 0.0 retention signal is ambiguous: it means either "this engine got every changed cell
    wrong" or "the held-out split had no changed cell to get right". The second is not a
    statement about the engine at all, and a reader who mistakes it for one draws the opposite
    conclusion. `true_changed_cells` is the denominator that separates them, so it is recorded
    on every round and on the retention record.

    It is a DIAGNOSTIC. It must never enter the comparison -- gating on it would change the
    shipped selection rule away from the one the counterfactual measured.

    MUTATION PROOF: drop `retention_true_changed` / `best_true_changed_cells` and this fails on
    the missing keys; wire the denominator into `is_best` (e.g. `retention_signal > best_signal
    or retention_true_changed > best_true_changed_cells`) and the informative arm's
    `best_round` moves off round 1.
    """

    # INFORMATIVE: the standard corpus changes cells on every transition.
    informative, _store, _proposer, _root = _run(
        monkeypatch, tmp_path, [GOOD_SRC, WORSE_SRC, WORST_SRC]
    )
    assert informative.engine_retention["signal_informative"] is True
    assert informative.engine_retention["best_round_true_changed_cells"] > 0
    assert informative.engine_retention["best_round"] == 1
    assert all(
        row["retention_signal_true_changed_cells"] > 0
        for row in informative.rounds
        if "retention_signal_true_changed_cells" in row
    )

    # UNINFORMATIVE: nothing in the corpus ever changes, so every round scores 0.0 and the tie
    # rule -- not evidence -- decides. The record must say so.
    uninformative, _store2, _proposer2, _root2 = _run(
        monkeypatch,
        tmp_path / "noop",
        [GOOD_SRC, WORSE_SRC, WORST_SRC],
        corpus=_all_noop_corpus(),
    )
    signals = [
        row["retention_signal_heldout_change_consistency"]
        for row in uninformative.rounds
        if "retention_signal_heldout_change_consistency" in row
    ]
    assert signals, "the loop must have scored at least one round"
    assert all(s == pytest.approx(0.0) for s in signals), (
        "an all-no-op split scores 0.0 for every engine, including a good one"
    )
    assert uninformative.engine_retention["signal_informative"] is False
    assert uninformative.engine_retention["best_round_true_changed_cells"] == 0


def test_the_informativeness_denominator_never_enters_the_comparison(monkeypatch, tmp_path):
    """The diagnostic must not become a second, unmeasured selection criterion.

    THIS TEST EXISTS BECAUSE A MUTANT SURVIVED. The first version of the diagnostic was mutation-
    tested with `is_best = ... or retention_true_changed > best_true_changed_cells` and NOTHING
    failed, which was initially rationalised as "an equivalent mutant -- `true_changed_cells`
    counts what REALITY changed, so it is a corpus constant". Reading
    `score_change_weighted_consistency` refuted that: the accumulation loop `continue`s BEFORE
    `true_changed_cells += n_changed_cells` when the engine RAISES or returns a wrong-shaped
    grid. So a crashing engine yields a SMALLER denominator, the mutant is reachable, and the
    suite had a genuine hole.

    The setup makes the two rules disagree:
      round 1  CRASHY_SRC -- exact on the moves it models, raises on every action-3 tick, so its
               denominator EXCLUDES those transitions and its consistency is high.
      round 2  WORSE_SRC  -- a clean no-op engine: consistency 0.0, but its denominator counts
               every changing transition, so it is strictly LARGER than round 1's.

    Shipped rule (rank on consistency alone) -> round 1 retained. Mutant (either-or) -> round 2
    wins on the denominator despite scoring 0.0 on the only signal that was ever measured.
    """

    result, store_text, proposer, _root = _run(
        monkeypatch, tmp_path, [CRASHY_SRC, WORSE_SRC, WORSE_SRC]
    )

    assert proposer.writes == [0, 1, 2]
    rows = [r for r in result.rounds if "retention_signal_heldout_change_consistency" in r]
    assert len(rows) >= 2

    crashy, clean = rows[0], rows[1]
    # The preconditions that make this test discriminating -- asserted, not assumed.
    assert (
        crashy["retention_signal_heldout_change_consistency"]
        > (clean["retention_signal_heldout_change_consistency"])
    ), "round 1 must win on the SIGNAL"
    assert (
        crashy["retention_signal_true_changed_cells"]
        < (clean["retention_signal_true_changed_cells"])
    ), "and must LOSE on the denominator, or the mutant would be unreachable here"

    # The shipped rule ranks on the signal alone.
    assert result.engine_retention["best_round"] == 1
    assert store_text == CRASHY_SRC
    assert [r.get("retained_as_best_engine") for r in rows][:2] == [True, False]


def test_retention_ranks_on_the_heldout_split_not_the_prefix(monkeypatch, tmp_path):
    """A prefix-memorising engine must NOT be retained over a genuinely general one.

    The round-1 engine reproduces the first six transitions exactly and no-ops on everything
    else: prefix change-consistency 1.0, held-out change-consistency 0.0. The round-2 engine is
    the general mover (held-out 0.8571).

    MUTATION PROOF: rank rounds on `prefix_change_consistency`, or on a consistency computed
    over the FULL transitions list rather than the held-out split, and round 1 wins -- which is
    selection on data the proposer was already shown, the classic overfit-peek. The test then
    fails on both the store assertion and the engine assertion.
    """

    transitions, _root = _corpus()
    overfit_src = _prefix_overfit_src(transitions)

    result, store_text, proposer, root = _run(
        monkeypatch, tmp_path, [overfit_src, GOOD_SRC, WORST_SRC]
    )

    assert proposer.writes == [0, 1, 2]
    assert result.engine_retention["best_round"] == 2
    assert store_text == GOOD_SRC
    assert np.array_equal(_predict(result.engine, root, 1), _true_next(root, 1))

    signals = [row.get("retention_signal_heldout_change_consistency") for row in result.rounds]
    assert signals[0] == pytest.approx(0.0), "the memoriser must score 0 on the held-out split"
    assert signals[1] == pytest.approx(0.8571, abs=1e-3)
    assert [row.get("retained_as_best_engine") for row in result.rounds] == [True, True, False]


def test_ties_keep_the_incumbent(monkeypatch, tmp_path):
    """A tie is no evidence to overwrite on -- and it is what the counterfactual replayed.

    WORSE and WORST both score 0.0 on the held-out split, so round 1 must survive rounds 2/3.

    MUTATION PROOF: relax the comparison from `>` to `>=` and round 3 displaces round 1,
    putting WORST_SRC on disk.
    """

    result, store_text, _proposer, _root = _run(
        monkeypatch, tmp_path, [WORSE_SRC, WORST_SRC, WORSE_SRC]
    )

    assert [row.get("retention_signal_heldout_change_consistency") for row in result.rounds] == [
        pytest.approx(0.0),
        pytest.approx(0.0),
        pytest.approx(0.0),
    ]
    assert result.engine_retention["best_round"] == 1
    assert store_text == WORSE_SRC


# =============================================================================================
# THE KILL SWITCH: default ON, and genuinely restorable to last-write-wins.
# =============================================================================================


def test_retention_is_on_by_default(monkeypatch):
    """DEFAULT-ON, stated honestly.

    The counterfactual bounded the downside directly: over 55 sliding 3-write windows retention
    HELPED 24, HURT 3, tied 28 (sign test on the 27 discordant windows, p = 4.9e-5, mean delta
    +0.0898). Default-on is defensible because 3-of-55 is the MEASURED downside, not because
    the downside is zero.

    MUTATION PROOF: flip the default to off and the first assertion fails.
    """

    monkeypatch.delenv("CARNOT_ARC_ENGINE_RETENTION", raising=False)
    assert engine_retention_enabled() is True
    monkeypatch.setenv("CARNOT_ARC_ENGINE_RETENTION", "0")
    assert engine_retention_enabled() is False
    monkeypatch.setenv("CARNOT_ARC_ENGINE_RETENTION", "1")
    assert engine_retention_enabled() is True


def test_disabling_retention_reproduces_exact_last_write_wins(monkeypatch, tmp_path):
    """The env flag must be a REAL A/B switch, not a label.

    With retention off, both defects come back exactly: round 3's source on disk AND round 3's
    engine returned. This is what makes `CARNOT_ARC_ENGINE_RETENTION=0` a usable control arm
    for anyone re-measuring the counterfactual against the shipped code.

    MUTATION PROOF: make the off-path still retain (e.g. leave the `not retention_on` term out
    of `is_best`) and both assertions below fail.
    """

    monkeypatch.setenv("CARNOT_ARC_ENGINE_RETENTION", "0")
    result, store_text, _proposer, root = _run(
        monkeypatch, tmp_path, [GOOD_SRC, WORSE_SRC, WORST_SRC]
    )

    assert store_text == WORST_SRC
    assert result.engine_retention["enabled"] is False
    assert result.engine_retention["reason"] == "disabled_by_env"
    predicted = _predict(result.engine, root, 1)
    assert not np.array_equal(predicted, _true_next(root, 1))
    assert int(predicted[0, 0]) == 1, "round 3's corner-writer must be what comes back"


# =============================================================================================
# EVIDENCE SAFETY.
# =============================================================================================


def test_retention_writes_only_inside_the_redirected_store(monkeypatch, tmp_path):
    """Retention must resolve `E3_DIR` at CALL time (REQ-ARC-WMTE-6016).

    MUTATION PROOF: bind the store path at import time (a module-level constant) and this test
    fails, because the monkeypatched `e3.E3_DIR` would be ignored and the real
    `results/arc_e3` -- which is measurement evidence -- would be written instead.
    """

    real_store = Path(__file__).resolve().parents[2] / "results" / "arc_e3"
    before = sorted(p.stat().st_mtime_ns for p in real_store.rglob("world_model.py"))

    _result, _store_text, _proposer, _root = _run(
        monkeypatch, tmp_path, [GOOD_SRC, WORSE_SRC, WORST_SRC]
    )

    assert (tmp_path / GAME / "world_model.py").exists()
    assert not (real_store / GAME).exists()
    after = sorted(p.stat().st_mtime_ns for p in real_store.rglob("world_model.py"))
    assert before == after, "no engine under results/arc_e3 may be touched by this test"
