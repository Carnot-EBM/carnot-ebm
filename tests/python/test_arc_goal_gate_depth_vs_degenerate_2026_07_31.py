"""REQ-ARC-WMTE-6047-D: a goal out of reach within `max_depth` is not a degenerate predicate.

THE INCIDENT (2026-07-30 gate-rejection audit, `results/outer_loop_arc_gate_forceadmit_20260730.json`).

The world-model trust gate rejected five of six games. On tn36 it rejected the ONE structurally
sound engine in the whole run -- heldout 8/8 byte-exact, `cell_recall` 1.0, `change_fidelity` 1.0,
zero invented cells, and when its plan was replayed against the real env, 61 of 61 steps matched
reality byte-for-byte -- with:

    kind:        degenerate_goal_predicate
    termination: queue_exhausted
    detail:      "the reachable set was searched exhaustively (frontier empty) and the goal was
                  never true, so this predicate is unreachable under this engine."

THAT DETAIL IS FALSE, and the record's own arithmetic says so. From the live cell
(`live_gate_records_per_cell["tn36___on"]["goal_satisfiability"]`):

    engine_calls               1480     = 40 popped-and-expanded nodes x 37 candidates
    reachable_grids_evaluated    41     = root + 40 new grids
    max_nodes                 20000     -> 1480 << 20000, so the BUDGET never fired
    max_depth                    40
    frontier_remaining            0

41 grids from 40 expansions means EVERY node produced a brand-new state: the search was a chain
that was still generating when it stopped. It stopped because the depth-40 node was popped and
discarded unexpanded by `if depth >= max_depth: continue`, after which the deque drained. tn36's
engine fills one cell of row 1 per click and its induced goal is `np.all(grid[1, 1:62] == 3)`.
Measured on the committed root-grid fixture, all 61 of those cells are still 9 at the root, so the
goal needs exactly 61 clicks against a cap of 40 -- which is why the gate never sees it.

(An earlier draft of this docstring said "6 already filled ... >= 55 more steps", taken from the
count of observed transitions in the engine's own comments. That is what the MODEL saw in its
induce prompt, not the state of the PLANNING ROOT, and the two are different grids. The measured
`first_true_depth == 61` -- asserted below -- is what settles it, and it was in the artifact the
whole time. Recorded rather than silently corrected: a number carried across from a summary
without being checked against the fixture is exactly the failure this file exists to document.)

The corroboration is direct: the same (engine, goal, root) triple run through `plan_in_model` at
`max_depth=200` FINDS a 61-action plan in 2226 nodes (`p2_stage1/tn36__on__s1.json`,
`deep_plan_found: true, deep_plan_length: 61`), while at the shipped `max_depth=40` it terminates
`queue_exhausted` at 1480 nodes with no plan. The predicate is reachable. It is not degenerate.

WHY THE DECISION IS NEVERTHELESS CORRECT, AND WHY THIS FIX IS ONLY THE LABEL.

`plan_in_model` is bounded by the SAME `max_depth=40` (`arc_executable_world_model.py:5371`), so a
goal unreachable within the cap is genuinely unreachable BY THE PLANNER THIS GATE GUARDS. Vetoing
on it is sound -- unlike vetoing on a spent NODE budget, where the planner has an independent
budget and the gate has learned nothing. So `satisfiable: False` stands, GOAL-REPAIR still fires
exactly where it fired before, and NAMING THIS KIND DOES NOT BY ITSELF MAKE tn36 PLANNABLE.

What a wrong reason costs is misdirection. `degenerate_goal_predicate` is the signal that tells
`refactor()` -- and every subsequent reader of the artifact -- "the induced win condition is junk".
On tn36 that pointed attention at a predicate the engine reaches at depth 61, and away from the
depth cap that actually stopped it. The audit that caught this had to reconstruct the arithmetic
by hand to disbelieve the gate's own `detail` string.

WHAT THESE TESTS PIN.

1. A search that discards nodes unexpanded at `max_depth` reports `goal_unreached_within_depth` +
   `termination: depth_capped`, never the degenerate kind.
2. The DECISION is unchanged: `satisfiable` is still False, so this cannot be read as widening a
   named quality gate. The same predicate is still vetoed; only its label moved.
3. A genuinely exhausted frontier -- nothing dropped at the cap -- still reports
   `degenerate_goal_predicate` + `queue_exhausted`. The degenerate verdict must still be earnable.
4. A budget-exhausted run reports the BUDGET kind and carries `depth_truncated_nodes == 0`, so
   the new counter cannot silently reclassify any case the sibling 2026-07-30 suite pins. (The
   two axes turn out to be mutually exclusive under this loop's BFS -- see that test's docstring
   for why the precedence ordering is therefore defensive rather than load-bearing today.)
5. The tn36 SHAPE end to end, on a scale model of its engine: a one-cell-per-click filler whose
   goal sits deeper than the cap, reproducing `reachable_grids_evaluated == 1 + expanded` (every
   node still generating) with `frontier_remaining == 0` and the budget untouched.
6. `execute_bounded_llm_reinduction` propagates the new kind into `row["skipped"]` and STILL
   routes it through GOAL-REPAIR -- the opposite of `goal_unreached_within_budget`, which is
   deliberately routed away from repair. That difference is the soundness argument above made
   executable.
"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from carnot.agentic import arc_llm_reinduction as reinduction


def _kind(result: dict) -> str:
    return str((result.get("counterexample") or {}).get("kind", ""))


# ------------------------------------------------------------------------------------------
# A SCALE MODEL OF tn36's ENGINE. The real one is 137 lines of the model narrating itself; its
# only actual behaviour is "action 6 turns the rightmost still-9 cell of row 1 into a 3, every
# other action is a no-op". That is a chain: exactly one new state per node regardless of how
# many candidates are offered, which is what produces 1480 calls / 41 grids.
# ------------------------------------------------------------------------------------------
_ROW = 1
_UNFILLED = 9
_FILLED = 3
_WIDTH = 16
_GOAL_SPAN = slice(1, _WIDTH - 2)  # mirrors tn36's `grid[1, 1:62]`


def _tn36_like_engine(grid: np.ndarray, action: int, _data: Any = None) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    if action != 6:
        return out
    for i in range(out.shape[1] - 1, -1, -1):
        if out[_ROW, i] == _UNFILLED:
            out[_ROW, i] = _FILLED
            break
    return out


def _tn36_like_goal(grid: np.ndarray) -> bool:
    return bool(np.all(np.asarray(grid)[_ROW, _GOAL_SPAN] == _FILLED))


def _tn36_like_board(prefilled: int = 0) -> np.ndarray:
    grid = np.zeros((4, _WIDTH), dtype=int)
    grid[_ROW, :] = _UNFILLED
    grid[_ROW, 0] = 5
    grid[_ROW, _WIDTH - 2 :] = 5
    for k in range(prefilled):
        grid[_ROW, _WIDTH - 3 - k] = _FILLED
    return grid


def _depth_to_goal(prefilled: int = 0) -> int:
    """How many clicks the goal actually needs -- the number the cap is compared against."""
    return int((_tn36_like_board(prefilled)[_ROW, _GOAL_SPAN] == _UNFILLED).sum())


# ------------------------------------------------------------------------------------------
# 1 + 2 + 5. The tn36 shape.
# ------------------------------------------------------------------------------------------


def test_depth_capped_search_reports_its_own_kind_and_still_vetoes() -> None:
    """THE tn36 CELL, reproduced: reachable goal, cap below it, budget untouched."""
    need = _depth_to_goal()
    cap = need - 3
    assert cap > 0

    result = reinduction._goal_satisfiability_check(
        engine=_tn36_like_engine,
        goal=_tn36_like_goal,
        start_grid=_tn36_like_board(),
        max_nodes=20_000,
        max_depth=cap,
    )

    # The label, which is what this commit changes.
    assert _kind(result) == "goal_unreached_within_depth", (
        "a search that threw away nodes at the depth cap did not search the reachable set "
        "exhaustively; calling it `degenerate_goal_predicate` accuses a reachable predicate"
    )
    assert result["termination"] == "depth_capped"
    assert result["depth_truncated_nodes"] >= 1
    # The false CLAIM is the damage, not the word -- and a bare `"exhaustively" not in detail`
    # is the negation-blind substring check CLAUDE.md's QA-layer discipline names by hand (the
    # replacement detail says "was NOT searched exhaustively", which such a check would reject).
    # So assert on the polarity.
    detail = result["counterexample"]["detail"]
    assert "NOT searched exhaustively" in detail
    assert "was searched exhaustively" not in detail, (
        "the false claim in the old detail string was the actual damage; it must not survive"
    )

    # The DECISION, which this commit does NOT change.
    assert result["satisfiable"] is False, (
        "this is a label fix. Flipping the veto would be widening a named quality gate, which "
        "is operator-authorisation territory"
    )

    # The tn36 arithmetic: budget nowhere near spent, frontier drained anyway.
    assert result["engine_calls"] < 20_000
    assert result["frontier_remaining"] == 0
    assert result["max_depth"] == cap


def test_the_same_predicate_is_reached_once_the_cap_allows_it() -> None:
    """What makes `degenerate` the wrong word: the goal IS reachable, just deeper than the cap."""
    need = _depth_to_goal()
    reachable = reinduction._goal_satisfiability_check(
        engine=_tn36_like_engine,
        goal=_tn36_like_goal,
        start_grid=_tn36_like_board(),
        max_nodes=20_000,
        max_depth=need + 5,
    )
    assert reachable["satisfiable"] is True
    assert reachable["first_true_depth"] == need


def test_every_expanded_node_produced_a_new_state_as_it_did_on_tn36() -> None:
    """The audit's arithmetic (41 grids = root + 40 expansions) is a property, not a fluke.

    A chain engine yields exactly one new grid per expanded node, so `reachable_grids_evaluated`
    must equal `1 + expanded`: the search was STILL GENERATING when it stopped, not running out
    of world. That is the signature the audit had to derive by hand in order to disbelieve the
    gate's own "searched exhaustively" claim, and it is the whole basis for the relabel.

    NB the audit's OTHER identity -- `engine_calls == expanded x candidates` (1480 = 40 x 37) --
    is deliberately NOT asserted here. It held on tn36 because a 64x64 board's connected-component
    structure is stable across the 40 steps, so `_model_candidates` returned the same 37 every
    time. On a 4x16 scale model the candidate count moves as row 1 fills, and asserting a constant
    would be pinning a fixture artifact rather than the property under test.
    """
    cap = 5
    result = reinduction._goal_satisfiability_check(
        engine=_tn36_like_engine,
        goal=_tn36_like_goal,
        start_grid=_tn36_like_board(),
        max_nodes=20_000,
        max_depth=cap,
    )
    expanded = cap  # depths 0..cap-1 are expanded; the depth-`cap` node is discarded
    assert result["reachable_grids_evaluated"] == expanded + 1
    assert result["depth_truncated_nodes"] == 1
    assert result["engine_calls"] >= expanded


# ------------------------------------------------------------------------------------------
# 3 + 4. The split must not swallow the two kinds it sits between.
# ------------------------------------------------------------------------------------------


def test_a_truly_exhausted_frontier_still_earns_the_degenerate_verdict() -> None:
    """Nothing dropped at the cap -> the old kind, unchanged. It must stay earnable."""

    def inert(grid: np.ndarray, _action: int, _data: Any = None) -> np.ndarray:
        return np.array(grid, dtype=int, copy=True)

    result = reinduction._goal_satisfiability_check(
        engine=inert,
        goal=lambda _g: False,
        start_grid=_tn36_like_board(),
        max_nodes=20_000,
        max_depth=40,
    )
    assert _kind(result) == "degenerate_goal_predicate"
    assert result["termination"] == "queue_exhausted"
    assert result["depth_truncated_nodes"] == 0


def test_budget_keeps_priority_and_the_depth_counter_never_steals_its_verdict() -> None:
    """The ordering that protects the sibling 2026-07-30 budget suite from this split.

    HONEST NOTE ON WHAT COULD NOT BE TESTED, because writing a fixture that "proves" precedence
    would have meant inventing a state the algorithm cannot enter. Under this gate's plain BFS the
    two axes are MUTUALLY EXCLUSIVE: depth truncation only begins once the frontier has reached
    the deepest layer, and a discarded node costs no engine calls, so no budget can be spent after
    the first drop. `depth_truncated_nodes > 0` and `engine_calls >= max_nodes` therefore cannot
    both hold at the end of a run. Several attempts to force it confirmed this rather than
    producing the fixture.

    So the `budget_exhausted`-first ordering is DEFENSIVE, not load-bearing today -- it matters
    only if this loop ever becomes best-first (which `arc_llm_reinduction.py`'s own
    `CARNOT_ARC_GRADED_GOAL_BIAS` caveat contemplates) and depth stops being monotone. What IS
    load-bearing and IS tested here: a budget-exhausted run reports the budget kind and carries a
    ZERO depth counter, so the new field cannot silently reclassify any existing budget case.
    """
    result = reinduction._goal_satisfiability_check(
        engine=_tn36_like_engine,
        goal=_tn36_like_goal,
        start_grid=_tn36_like_board(),
        max_nodes=3,
        max_depth=40,
    )
    assert _kind(result) == "goal_unreached_within_budget"
    assert result["termination"] == "budget_exhausted"
    assert result["depth_truncated_nodes"] == 0

    # And the ordering itself, read off the source: `budget_exhausted` is computed first and the
    # depth flag is explicitly gated on `not budget_exhausted`.
    import inspect

    src = inspect.getsource(reinduction._goal_satisfiability_check)
    assert "depth_truncated = (not budget_exhausted) and depth_truncated_nodes > 0" in src


def test_max_depth_zero_discards_the_root_and_says_so() -> None:
    """The degenerate case of the degenerate case: nothing at all was searched."""
    result = reinduction._goal_satisfiability_check(
        engine=_tn36_like_engine,
        goal=_tn36_like_goal,
        start_grid=_tn36_like_board(),
        max_nodes=20_000,
        max_depth=0,
    )
    assert _kind(result) == "goal_unreached_within_depth"
    assert result["depth_truncated_nodes"] == 1
    assert result["engine_calls"] == 0
    assert result["reachable_grids_evaluated"] == 1


# ------------------------------------------------------------------------------------------
# 6. The caller. This is the half that decides whether the new kind is inert or harmful.
# ------------------------------------------------------------------------------------------


def test_depth_capped_kind_reaches_the_row_and_still_routes_through_goal_repair(
    monkeypatch,
) -> None:
    """`goal_unreached_within_depth` must behave like `degenerate`, NOT like `within_budget`.

    The budget kind is deliberately routed AWAY from GOAL-REPAIR: repair substitutes a looser
    "strictly fuller than root" proxy, and doing that on the strength of a spent compute budget
    turns a ceiling into a goal rewrite. The depth kind is different in exactly the way that
    matters -- `plan_in_model` shares the cap, so the veto is EARNED -- and a reachable-but-too-deep
    goal is precisely the case where a reachable proxy is the productive fallback. So repair
    must still fire, and `goal_undecided_within_budget` must stay False.

    An earlier draft of this test built a FAKE gate result and asserted things about IT, which
    asserted nothing whatsoever about the shipped code -- caught in adversarial self-review and
    recorded rather than quietly swapped. This one drives the real
    `execute_bounded_llm_reinduction`, modelled on the sibling suite's
    `test_budget_starved_gate_does_not_rewrite_the_goal_end_to_end` so the two are directly
    comparable: same harness, opposite expected routing.

    NB the caller invokes the gate with NO `max_nodes`/`max_depth`, so the fixture has to be deep
    enough to trip the SHIPPED `max_depth=40` rather than a value this test could pass in.
    """
    from carnot.agentic.arc_executable_world_model import Transition
    from carnot.agentic.arc_llm_reinduction import execute_bounded_llm_reinduction

    # A chain world whose goal sits at depth 45, against the shipped cap of 40 -- reachable, but
    # not within the cap. A chain yields one new state per node, so the search costs ~40 x
    # candidates engine calls, nowhere near the 20,000 budget: the ONLY thing that can classify
    # this round is the depth axis. (Same shape as tn36, which needs 61 against the same 40.)
    target_depth = 45
    root = np.zeros((1, 64), dtype=int)

    def engine(grid: np.ndarray, action: int, data: Any = None) -> np.ndarray:
        out = np.array(grid, dtype=int, copy=True)
        if int(action) == 1:
            filled = int((out[0] != 0).sum())
            if filled < out.shape[1]:
                out[0, filled] = 5
        return out

    def deep_goal(grid: np.ndarray) -> bool:
        return bool(int((np.asarray(grid)[0] == 5).sum()) == target_depth)

    class _Proposer:
        model = "fixture"

        def induce(self, _game, _transitions, _cell):  # noqa: ANN001,ANN202
            return True, "candidate"

        def refactor(self, _game, _counterexample):  # noqa: ANN001,ANN202
            return True, "refined"

    transitions = [
        Transition(
            grid=root.copy(),
            action=1,
            data=None,
            next_grid=engine(root, 1, None),
            level_before=1,
            level_after=1,
        )
    ]

    # SPY on GOAL-REPAIR rather than inferring from the label. "Repair still fires" is the actual
    # claim; a skip label consistent with it is weaker evidence, because the round can also reach
    # that label by repair returning None -- which is in fact what happens here (the exemplar-derived
    # fallback is itself past the cap). Counting the call separates "repair was attempted and could
    # not help" from "repair was never reached", and only the second would be the behaviour change
    # this commit promises it is not making.
    repair_calls: list[dict] = []
    _real_repair = reinduction._repair_degenerate_goal

    def _spy_repair(**kwargs: Any):  # noqa: ANN202
        repair_calls.append(kwargs)
        return _real_repair(**kwargs)

    monkeypatch.setattr(reinduction, "_repair_degenerate_goal", _spy_repair)

    result = execute_bounded_llm_reinduction(
        game="depth_fixture",
        transitions=transitions,
        cell=1,
        root_grid=root,
        proposer=_Proposer(),
        candidate_provider=lambda e, g: [("c", e, g)],
        load_engine=lambda _game: (engine, deep_goal),
        plan_in_model=lambda e, g, s: None,
        max_rounds=1,
        # An exemplar IS available, so GOAL-REPAIR is ARMED. That is the point of the test: the
        # depth kind must reach repair, where the budget kind must not.
        previous_level_complete_grid=np.full((1, 64), 5, dtype=int),
    )

    row0 = result.rounds[0]
    assert row0["goal_satisfiability"]["termination"] == "depth_capped", (
        "the fixture must actually exercise the depth axis, not some other stop"
    )
    assert row0["goal_satisfiability"]["depth_truncated_nodes"] >= 1

    # THE ROUTING. Not routed like the budget kind...
    assert row0["goal_undecided_within_budget"] is False, (
        "depth truncation is NOT budget exhaustion; treating it as such would silently stop "
        "GOAL-REPAIR firing on a case where the veto is earned"
    )
    # ...but exactly like `degenerate_goal_predicate` was: GOAL-REPAIR is ATTEMPTED. This is the
    # load-bearing assertion of the whole test.
    assert len(repair_calls) == 1, (
        "GOAL-REPAIR must still be reached on a depth-capped gate. If this drops to 0 the depth "
        "kind has been routed like the budget kind, which is a real behaviour change and was "
        "deliberately not made (mutation M6)"
    )
    # It returns None here -- the exemplar-derived fallback is itself past the cap -- so the round
    # skips CARRYING THE NEW KIND rather than substituting a goal. Both halves matter.
    assert row0.get("skipped") == "goal_unreached_within_depth"
    assert "goal_repaired" not in row0

    # And the audit trail names the depth case rather than accusing the predicate.
    kinds = [str(c.get("kind", "")) for c in result.counterexamples]
    if kinds:
        assert "degenerate_goal_predicate" not in kinds, (
            "a depth-capped search must never be recorded as a disproved predicate"
        )

    # The routing predicate itself, read off the shipped source: the budget branch is keyed on the
    # LITERAL kind string, so a third kind cannot fall into it by accident.
    import inspect

    src = inspect.getsource(reinduction.execute_bounded_llm_reinduction)
    assert "goal_undecided_within_budget = not round_goal_satisfiable and (" in src
    assert '== "goal_unreached_within_budget"' in src, (
        "if this equality ever becomes a substring/prefix test, `goal_unreached_within_depth` "
        "would start being routed away from GOAL-REPAIR without anyone deciding that"
    )
    assert '"goal_unreached_within_depth"' not in src, (
        "the depth kind must NOT acquire its own skip-without-repair branch without the "
        "measurement to justify it"
    )


def test_agent_plain_path_allow_list_discloses_the_depth_kind() -> None:
    """The plain path flattens any unrecognised kind back to `degenerate_goal_predicate`.

    That allow-list is a whitelist, so adding a kind to the gate without adding it here would
    silently reinstate the exact mislabel this commit removes -- while the gate's own artifact
    field said something different. Behaviour is unchanged either way (all listed kinds skip);
    this is purely about what gets recorded.
    """
    import inspect

    from carnot.agentic import arc_competition_agent as aca

    src = inspect.getsource(aca)
    marker = '"goal_unreached_within_depth",'
    assert marker in src, "the plain path would flatten the depth kind back to `degenerate`"


# ------------------------------------------------------------------------------------------
# THE ORIGIN INCIDENT ITSELF, against the REAL engine and the REAL root grid.
#
# Everything above is a scale model. This is the actual tn36 `on` cell the 2026-07-30 audit
# mislabelled, pinned so the claim survives the session that measured it.
#
# WHY THESE TWO FILES ARE COMMITTED. The audit could not speak for lp85's rejected engine at all,
# because its source lived in a per-cell temp store that no longer exists -- it had to reason from
# a RECONSTRUCTION and said so. tn36's engine was one `rm -rf /tmp` from the same fate, and its
# root grid cost a 38-second env replay to recover every time anyone wanted to ask a question
# about it. Both are deterministic constants; there is no reason to keep re-deriving them.
#
#   tn36_on_world_model.py.frozen  md5 6d96491f80bec0319828ba1a04f5841e, 5256 B -- byte-identical
#     to the engine recorded at `live_gate_records_per_cell["tn36___on"].engine_retention
#     .store_path`, i.e. the exact engine the gate rejected.
#   tn36_on_root_grid.npy          sha256[:16] f328c951a03d248d, 64x64 -- byte-identical to the
#     `root_grid_sha256` the audit's Stage 1 recorded for the same cell.
#
# Both hashes are asserted below, so a fixture swap fails loudly instead of quietly changing what
# the test is about. FROZEN: read, never written -- same contract as `load_origin_fixture_engine`.
# ------------------------------------------------------------------------------------------

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "arc_goal_gate_depth_tn36"
_TN36_ENGINE_MD5 = "6d96491f80bec0319828ba1a04f5841e"
_TN36_ROOT_SHA = "f328c951a03d248d"


def _load_tn36_cell(tmp_path):
    """Load the frozen engine through the SHIPPED loader, from a copy, never the fixture itself."""
    from carnot.agentic import arc_executable_world_model as e3

    src = (_FIXTURES / "tn36_on_world_model.py.frozen").read_bytes()
    assert hashlib.md5(src).hexdigest() == _TN36_ENGINE_MD5, (
        "the frozen engine is not the one the 2026-07-30 audit rejected; this test would be "
        "asserting about a different artifact"
    )
    dst = tmp_path / "tn36"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "world_model.py").write_bytes(src)
    engine, is_done = e3._load_engine_from(tmp_path, "tn36")

    root = np.load(_FIXTURES / "tn36_on_root_grid.npy")
    assert hashlib.sha256(root.tobytes()).hexdigest()[:16] == _TN36_ROOT_SHA
    return engine, is_done, root


def test_tn36_the_real_cell_reports_depth_not_degeneracy(tmp_path) -> None:
    """THE ORIGIN INCIDENT. Real engine, real root, gate at its live defaults.

    Reproduces the audit's record to the digit -- 1480 engine calls, 41 grids, budget 20000
    untouched, frontier drained -- and asserts the verdict it now carries.
    """
    engine, is_done, root = _load_tn36_cell(tmp_path)
    result = reinduction._goal_satisfiability_check(engine=engine, goal=is_done, start_grid=root)

    # The audit's arithmetic, reproduced exactly.
    assert result["engine_calls"] == 1480
    assert result["reachable_grids_evaluated"] == 41
    assert result["max_nodes"] == 20000
    assert result["max_depth"] == 40
    assert result["frontier_remaining"] == 0
    assert result["engine_calls"] < result["max_nodes"], (
        "if the budget ever fires here the cell is no longer the one under audit"
    )

    # The label. Before this commit the branch was `budget ? within_budget : degenerate`, and
    # with the budget untouched that is `degenerate_goal_predicate` -- the mislabel.
    assert _kind(result) == "goal_unreached_within_depth"
    assert result["termination"] == "depth_capped"
    assert result["depth_truncated_nodes"] == 1

    # The decision, unchanged. This is a relabel, not a relaxation.
    assert result["satisfiable"] is False


def test_tn36_goal_is_reachable_at_depth_61_which_is_why_degenerate_was_wrong(tmp_path) -> None:
    """The predicate the gate called degenerate is reached by the very same engine at depth 61.

    And -- stated because it is the honest scope of the whole fix -- the 61 is still past the
    shipped `max_depth=40`, so `plan_in_model` finds nothing at production settings. Naming the
    kind correctly does NOT by itself make tn36 plannable. It stops the next fix being aimed at
    a goal predicate that was never the problem.
    """
    from carnot.agentic import arc_executable_world_model as e3

    engine, is_done, root = _load_tn36_cell(tmp_path)

    # The arithmetic, pinned against the fixture rather than against a summary. tn36's induced
    # goal is `np.all(grid[1, 1:62] == 3)`, its engine fills one such cell per click, and at the
    # PLANNING ROOT none of them are filled -- so the depth the goal sits at is forced to be 61,
    # and any claim of a smaller number is checkable and wrong.
    goal_span = root[1, 1:62]
    assert goal_span.size == 61
    assert int((goal_span == 3).sum()) == 0, "no cell of the goal span is pre-filled at the root"
    assert int((goal_span == 9).sum()) == 61

    lifted = reinduction._goal_satisfiability_check(
        engine=engine, goal=is_done, start_grid=root, max_nodes=300_000, max_depth=200
    )
    assert lifted["satisfiable"] is True
    assert lifted["first_true_depth"] == 61
    assert lifted["first_true_depth"] == int((goal_span != 3).sum()), (
        "one click per unfilled cell -- the depth IS the count, so the two must agree"
    )

    assert e3.plan_in_model(engine, is_done, root, max_nodes=300_000, max_depth=40) in (None, [])
    deep = e3.plan_in_model(engine, is_done, root, max_nodes=300_000, max_depth=80)
    assert deep is not None and len(deep) == 61
