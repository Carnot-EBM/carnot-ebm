"""Tests for wiring ``HazardMovePruner`` into the SCORED agent (REQ-ARC-WMTE-5970).

WHY THIS FILE EXISTS. Every pruning lever this project has measured was CLICK-side, while a corpus
census of the shipped scored agent found the action mass is NAV-side: 17 of 25 public games' modal
repeated action is nav/keyboard and 6 games issue zero clicks in 2000 actions.
``arc_hazard_pruner.HazardMovePruner`` was the one nav-side pruner the project already owned, and it
was reachable ONLY from ``scripts/arc_loop_solve.py`` (the offline dev twin) via
``arc_solver_kit.OfflineSolver`` -- ``arc_competition_agent`` referenced it in a single prose comment
and never imported it. So its one measured result (tu93 L3, states_expanded 2947 -> 2859) said
nothing about the scored path.

WHAT THESE TESTS PROTECT, in order of how badly the project has been burned by each:

1. THE OBSERVE CHANNEL MUST BE LIVE ON A BARE EXPLORER (``test_..._observe_channel_survives_a_bare_
   explorer``). This is the failure the project has already made twice. ``awaiting["previous_frame"]``
   is ``graph[origin]["frame"]``, and node frames are RETAINED only when one of nine unrelated
   optional components is attached; ``awaiting["grid"]`` reads the SAME field, so an
   ``or o.get("grid")`` fallback rescues nothing. On the exp5836 ``CarnotAgentPolicy`` harness 0 of
   122 graph nodes carried ``previous_frame``, so a pruner reported ``observed=0 pruned=0`` -- a
   clean, zero-error, byte-identical NULL that was pure harness artifact. An existing sibling test
   PASSED against that dead channel because it hand-injects ``previous_frame``. The bare-explorer
   test below does NOT hand-inject it: it reproduces ``_serve``'s construction faithfully (both
   fields resolved from the graph, hence None) and asserts the pruner still observes.

2. A FULLY-PRUNED NODE MUST NEVER BE EMPTIED (``test_..._never_empty_guard``). An empty
   ``untested`` list makes ``_node_has_open_tier`` False, which can drive ``next_move`` to
   ``explored_out = True`` and END THE RUN EARLY -- a mechanical null indistinguishable from a
   behavioural one.

3. THE PURE HAZARD LOGIC MUST ACTUALLY LEARN (``test_pruner_learns_...``). The pre-existing prune
   test injects ``p._model = _StubModel()``, so the suite proved delegation but never proved the
   pruner can FIT. That matters here because two independent censuses say the model fits on 0 of 25
   / 1 of 15 public games -- if the fit path were also broken, a corpus null would be doubly
   uninterpretable. The test below drives the REAL ``InducedNavWorldModel`` /
   ``HazardAwareNavWorldModel`` fit from constructed charger transitions, no stub.

4. DEFAULT-OFF PARITY (``test_..._default_off_...``). The lever ships default-OFF; nothing about the
   scored path may move until an operator flips it.

Spec refs: REQ-ARC-WMTE-5970, SCENARIO-ARC-WMTE-5970-LIVE-WIRING-CANDIDATES,
SCENARIO-ARC-WMTE-5970-LIVE-WIRING-OBSERVE, SCENARIO-ARC-WMTE-5970-NEVER-EMPTY,
SCENARIO-ARC-WMTE-5970-FIRE-COUNTERS, SCENARIO-ARC-WMTE-5970-DEFAULT-OFF-PARITY.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot.agentic.arc_competition_agent import (
    SUBMITTED_AGENT_CONFIG,
    SUBMITTED_HAZARD_MOVE_PRUNER_ENABLED,
    CarnotAgentPolicy,
    E3AgentPolicy,
    StepwiseExplorer,
)
from carnot.agentic.arc_hazard_pruner import HazardMovePruner, coerce_hazard_move_pruner

# --------------------------------------------------------------------------------------------
# A fully-known ground-truth nav mechanic with a line CHARGER, independent of the fitter.
# Deliberately self-contained (rather than imported from test_arc_nav_world_model) so this file
# cannot be silently broken by an edit to another test module's private fixtures.
#   colour 5 = wall/background, 2 = door, 0 = floor, 4 = avatar (1x1), 14 = goal, 8 = charger.
# --------------------------------------------------------------------------------------------
WALL, DOOR, FLOOR, AV, GOAL, HAZ = 5, 2, 0, 4, 14, 8
STEP = 2
_DIRS = {1: (-STEP, 0), 2: (STEP, 0), 3: (0, -STEP), 4: (0, STEP)}


def _grid(av_rc, *, mid=None, mid_color=None, dest=None, dest_color=None) -> np.ndarray:
    g = np.full((7, 7), WALL, dtype=int)
    g[av_rc] = AV
    if mid is not None:
        g[mid] = mid_color
    if dest is not None:
        g[dest] = dest_color
    return g


def _mk(av, action, outcome):
    """A (g0, action, g1, level_before, level_after) transition with a known outcome: 'move' (door
    ahead -> avatar advances), 'block' (wall ahead -> avatar stays), 'levelup' (goal ahead)."""
    dy, dx = _DIRS[action]
    r, c = av
    mid = (r + dy // 2, c + dx // 2)
    dest = (r + dy, c + dx)
    if outcome == "block":
        g0 = _grid(av, mid=mid, mid_color=WALL)
        return (g0, action, g0.copy(), 0, 0)
    dest_color = GOAL if outcome == "levelup" else FLOOR
    g0 = _grid(av, mid=mid, mid_color=DOOR, dest=dest, dest_color=dest_color)
    g1 = _grid(dest, mid=mid, mid_color=DOOR)
    g1[av] = FLOOR
    return (g0, action, g1, 0, 1 if outcome == "levelup" else 0)


def _safe_transitions() -> list:
    """Every direction with a move and a wall-block, plus one goal level-up so the goal colour is
    learned from the level-up signal rather than a heuristic fallback. NO deaths here."""
    tr = []
    for action in (1, 2, 3, 4):
        tr.append(_mk((3, 3), action, "move"))
        tr.append(_mk((3, 3), action, "block"))
    tr.append(_mk((3, 3), 4, "move"))
    tr.append(_mk((3, 1), 4, "move"))
    tr.append(_mk((1, 3), 2, "move"))
    tr.append(_mk((3, 3), 4, "levelup"))
    return tr


def _charger_grid(av_col: int) -> np.ndarray:
    """A 7x9 field: the avatar on row 3 at `av_col`, a colour-8 charger blob centred at (3,5)."""
    g = np.full((7, 9), WALL, dtype=int)
    g[3, av_col] = AV
    g[3, 4:7] = HAZ
    g[2, 5] = HAZ
    g[4, 5] = HAZ
    return g


def _death_transition():
    """The avatar approaches the charger along its row; the charger CHARGES left to intercept and
    the avatar is REMOVED (absent in g1). This is the signature `_death_labels` keys on."""
    g0 = _charger_grid(1)
    g1 = np.full((7, 9), WALL, dtype=int)  # avatar REMOVED
    g1[3, 2:5] = HAZ
    g1[2, 3] = HAZ
    g1[4, 3] = HAZ
    return (g0, 4, g1, 0, 0)


def _hazard_transitions(n_deaths: int = 3) -> list:
    tr = list(_safe_transitions())
    tr.extend([_death_transition()] * n_deaths)
    return tr


def _feed(pruner: HazardMovePruner, transitions: list) -> None:
    for g0, action, g1, lb, la in transitions:
        pruner.observe(g0, {"action": int(action)}, g1, leveled_up=bool(la > lb))


# ============================================================================================
# 1. THE PURE HAZARD LOGIC -- real fits, no stub model
# ============================================================================================


def test_pruner_learns_hazard_from_observed_deaths_and_prunes_only_the_lethal_move() -> None:
    """REQ-ARC-WMTE-5970: the pruner fits a REAL hazard model from observed avatar-removal deaths
    and then prunes the move toward the charger while leaving the move away from it alone.

    This is the end-to-end fit path -- ``InducedNavWorldModel.fit`` for the avatar/goal/wall
    colours, then ``HazardAwareNavWorldModel.fit`` per lethal rung, then the in-sample
    trust/specificity gate. The pre-existing prune test injects ``p._model``, so nothing in the
    suite previously proved this path works at all.
    """

    pruner = HazardMovePruner(np.asarray, refit_every=1000)  # refit only when we say so
    _feed(pruner, _hazard_transitions(n_deaths=3))
    assert pruner.stats()["model_fitted"] is False, "no fit until the refit cadence fires"

    pruner._fit()
    stats = pruner.stats()
    assert stats["n_deaths"] == 3, f"the death predicate must see all 3 deaths, got {stats}"
    assert stats["model_fitted"] is True, f"a real hazard fit must be reached, got {stats}"
    assert stats["lethal_mode"] in ("toward", "omni")
    assert stats["trust"] >= 0.9 and stats["specificity"] >= 0.98

    probe = _charger_grid(1)  # avatar at (3,1), charger centred at (3,5) on the same row
    assert pruner.should_prune(probe, {"action": 4}) is True, "moving toward the charger is lethal"
    assert pruner.should_prune(probe, {"action": 3}) is False, "moving away from it is safe"
    assert pruner.stats()["pruned"] == 1, "only the lethal probe may be counted as a prune"


def test_pruner_refuses_to_fit_below_min_deaths_even_with_real_deaths_present() -> None:
    """The evidence bar is load-bearing, not decorative: with 2 real deaths and min_deaths=3 the
    pruner must stay unfitted and prune NOTHING, so a game with a rare one-off death cannot have
    its search shaped by a single sample."""

    pruner = HazardMovePruner(np.asarray, refit_every=1000, min_deaths=3)
    _feed(pruner, _hazard_transitions(n_deaths=2))
    pruner._fit()
    stats = pruner.stats()
    assert stats["n_deaths"] == 2
    assert stats["model_fitted"] is False
    assert pruner.should_prune(_charger_grid(1), {"action": 4}) is False
    assert stats["pruned"] == 0


def test_pruner_refuses_to_fit_when_the_specificity_gate_cannot_be_met() -> None:
    """A rung that cannot clear the false-positive bar must be REJECTED, leaving the search
    unmodified. Raising ``min_specificity`` above 1.0 makes the bar unreachable by construction,
    which isolates "is the gate actually consulted" from "is this fixture's rung good".

    Over-masking a safe move can break a solve while under-pruning only costs efficiency, so this
    asymmetry is the entire reason the gate exists.
    """

    pruner = HazardMovePruner(np.asarray, refit_every=1000, min_specificity=1.01)
    _feed(pruner, _hazard_transitions(n_deaths=3))
    pruner._fit()
    assert pruner.stats()["n_deaths"] == 3, "the deaths must still be COUNTED"
    assert pruner.stats()["model_fitted"] is False, "but no rung may pass an unreachable bar"
    assert pruner.should_prune(_charger_grid(1), {"action": 4}) is False


def test_pruner_handles_degenerate_and_empty_inputs_without_raising() -> None:
    """Degenerate cases the live path really produces: a click label (no nav action), mismatched
    grid shapes across a level transition, a flattened 1-D frame, an empty grid, and a grid_of that
    raises. None may raise, and none may cause a prune."""

    pruner = HazardMovePruner(np.asarray, refit_every=1000, nav_actions_only=True)
    g = np.zeros((5, 5), dtype=int)

    pruner.observe(g, {"action": 1}, np.zeros((7, 7), dtype=int), leveled_up=False)
    assert pruner.observed == 0, "a shape-mismatched transition must be dropped"

    flat = np.zeros(25, dtype=int)  # 5x5 flattened -- _g2d must square it back up
    pruner.observe(flat, {"action": 1}, flat, leveled_up=False)
    assert pruner.observed == 1, "a flattened square frame must be recovered, not dropped"

    empty = np.zeros((0, 0), dtype=int)
    pruner.observe(empty, {"action": 1}, empty, leveled_up=False)
    assert pruner.should_prune(empty, {"action": 1}) is False

    def _raises(_frame: Any) -> np.ndarray:
        raise RuntimeError("grid_of blew up")

    boom = HazardMovePruner(_raises)
    boom.observe(g, {"action": 1}, g, leveled_up=False)
    assert boom.observed == 0
    assert boom.should_prune(g, {"action": 1}) is False


def test_live_click_labels_decode_as_nav_action_6_which_is_why_nav_actions_only_exists() -> None:
    """DEFECT FOUND BY THIS TEST FILE (2026-07-26), pinned so it cannot silently reappear.

    ``arc_hazard_pruner``'s module docstring claims the pruner "NO-OPS ... so it is safe to enable
    for ANY game -- a non-nav / non-hazard game never fits a hazard model", and
    ``_default_action_of_label``'s docstring claimed it "returns None for non-nav labels
    (click/paint games)". That second claim is FALSE for the label shape both live consumers
    actually pass. ``StepwiseExplorer._ingest`` builds ``{"action": int(o["action"]), "data":
    o.get("data")}`` and ``arc_solver_kit`` passes the same shape, so a click arrives as
    ``{"action": 6, "data": {"x": .., "y": ..}}`` and decodes to the int 6 -- it is BUFFERED as a
    keyboard-nav transition. On a click-heavy game (one public game issues 1043 clicks in ~2000
    actions) the buffer that ``InducedNavWorldModel.fit`` sees is then dominated by pointer rows and
    the fitter is asked to learn a spatial displacement for action 6, which can only degrade the
    avatar/displacement fit the death predicate depends on.

    Only the coordinate-ONLY label shape was ever filtered. This test asserts BOTH halves: the
    unfiltered default (preserved so the offline twin's published tu93 A/B stays bit-reproducible)
    and the ``nav_actions_only=True`` opt-in the scored-path coercion uses.
    """

    g = np.zeros((5, 5), dtype=int)
    live_click = {"action": 6, "data": {"x": 1, "y": 1}}

    unfiltered = HazardMovePruner(np.asarray, refit_every=1000)
    unfiltered.observe(g, live_click, g, leveled_up=False)
    assert unfiltered.observed == 1, (
        "the DEFAULT still buffers a live click as nav action 6 -- this is the defect, pinned "
        "deliberately rather than fixed in place, because this default is shared with the offline "
        "dev twin whose published measurement must stay bit-reproducible"
    )
    assert unfiltered.stats()["nav_actions_only"] is False
    assert unfiltered.stats()["clicks_skipped"] == 0

    filtered = HazardMovePruner(np.asarray, refit_every=1000, nav_actions_only=True)
    filtered.observe(g, live_click, g, leveled_up=False)
    filtered.observe(g, '{"action": 6, "data": {"x": 2, "y": 2}}', g, leveled_up=False)
    filtered.observe(g, {"action": 2, "data": None}, g, leveled_up=False)
    assert filtered.observed == 1, "only the genuine nav transition may be buffered"
    assert filtered.stats()["clicks_skipped"] == 2, "and both click shapes must be counted as skips"
    # The prune side was never the risk (action 6 is absent from any fitted displacement map, so
    # is_lethal returns False for it) but the filter must hold there too, cheaply and explicitly.
    filtered._model = SimpleNamespace(is_lethal=lambda grid, action: True)
    assert filtered.should_prune(g, live_click) is False, "a click may never be hazard-pruned"
    assert filtered.should_prune(g, {"action": 2, "data": None}) is True


def test_the_scored_path_coercion_opts_into_the_nav_only_filter() -> None:
    """The scored path must get the filtered construction; that is the whole point of the opt-in."""

    built = coerce_hazard_move_pruner(True)
    assert isinstance(built, HazardMovePruner)
    assert built.nav_actions_only is True, (
        "the live scored explorer sees click labels; an unfiltered pruner there would pollute the "
        "nav fit with pointer rows"
    )
    explorer = StepwiseExplorer(hazard_move_pruner=True)
    assert explorer.hazard_move_pruner_diagnostics()["nav_actions_only"] is True


def test_stats_reports_the_fields_a_fire_counter_verdict_needs() -> None:
    """A zero prune count has three distinct causes and an A/B must be able to tell them apart, so
    stats() must expose the evidence, the fit outcome AND the gate thresholds it was judged by."""

    stats = HazardMovePruner(np.asarray).stats()
    for key in (
        "observed",
        "pruned",
        "n_deaths",
        "model_fitted",
        "lethal_mode",
        "transitions_buffered",
        "min_deaths",
        "min_trust",
        "min_specificity",
        "nav_actions_only",
        "clicks_skipped",
    ):
        assert key in stats, f"stats() must report {key}"
    assert stats["verifier_is_oracle"] is False, (
        "the pruner is a LEARNED hazard predictor, not the executable oracle that defines "
        "correctness -- per the Circularity/Oracle-Distinctness discipline"
    )


# ============================================================================================
# 2. THE COERCION
# ============================================================================================


def test_coerce_hazard_move_pruner_none_false_true_and_instance() -> None:
    assert coerce_hazard_move_pruner(None) is None
    assert coerce_hazard_move_pruner(False) is None
    default = coerce_hazard_move_pruner(True)
    assert isinstance(default, HazardMovePruner)
    instance = HazardMovePruner(np.asarray)
    assert coerce_hazard_move_pruner(instance) is instance
    # anything else falls through to None, matching every sibling coercion's strict-isinstance
    # discipline (a stray string or mapping must not silently become an enabled lever).
    assert coerce_hazard_move_pruner({"mode": "on"}) is None
    assert coerce_hazard_move_pruner("on") is None


# ============================================================================================
# 3. THE LIVE PRUNE HOOK IN StepwiseExplorer._candidates
# ============================================================================================


class _AlwaysPrune:
    """Prunes every row. Isolates the never-empty guard from the real model's fit logic."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict]] = []

    def should_prune(self, frame: Any, label: Any) -> bool:
        self.calls.append((frame, label))
        return True

    def stats(self) -> dict:
        return {"observed": 0, "pruned": len(self.calls), "model_fitted": True}


class _PruneOneAction:
    """Prunes exactly one action id -- the realistic shape (a hazard model condemns a direction)."""

    def __init__(self, action: int) -> None:
        self.action = int(action)
        self.calls: list[tuple[Any, dict]] = []

    def should_prune(self, frame: Any, label: Any) -> bool:
        self.calls.append((frame, label))
        return int(label["action"]) == self.action

    def stats(self) -> dict:
        return {"observed": 0, "pruned": 0, "model_fitted": True}


def _bare_explorer(**kwargs: Any) -> StepwiseExplorer:
    """A bare explorer with no frame-retaining optional component attached -- the configuration
    that exposed the dead observe channel."""

    return StepwiseExplorer(
        online_discriminative=False,
        navigation_cost_tiebreak=False,
        frame_change_scorer=None,
        action_effect_expansion_prior=False,
        **kwargs,
    )


def _frame(actions: list[int], grid: np.ndarray | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        frame=np.array([[1]], dtype=np.int16) if grid is None else grid,
        available_actions=list(actions),
    )


def test_scenario_arc_wmte_5970_candidates_consults_the_pruner_and_drops_its_rows() -> None:
    """SCENARIO-ARC-WMTE-5970-LIVE-WIRING-CANDIDATES: ``_candidates`` calls
    ``hazard_move_pruner.should_prune`` per row with the node's OWN frame and uses the result."""

    pruner = _PruneOneAction(2)
    explorer = _bare_explorer()
    explorer.hazard_move_pruner = pruner
    frame = _frame([1, 2, 3, 4])

    rows = explorer._candidates(frame, path=[])

    assert pruner.calls, "the pruner must actually be consulted"
    assert all(seen_frame is frame for seen_frame, _ in pruner.calls), (
        "the antecedent must be the node's own frame -- that is what should_prune judges"
    )
    assert 2 not in {int(r["action"]) for r in rows}, "the condemned action must be dropped"
    assert {1, 3, 4} <= {int(r["action"]) for r in rows}, "every other action must survive"
    assert explorer._hazard_rows_pruned == 1
    assert explorer._hazard_all_pruned_nodes == 0
    assert explorer._hazard_prune_errors == 0


def test_scenario_arc_wmte_5970_never_empty_guard_keeps_one_row() -> None:
    """SCENARIO-ARC-WMTE-5970-NEVER-EMPTY: when the pruner condemns EVERY row, exactly one row
    survives and the event is counted.

    Load-bearing rather than defensive: an empty ``untested`` list makes ``_node_has_open_tier``
    False, which can drive ``next_move`` to ``explored_out = True`` and end the run early -- a
    mechanical null that looks exactly like a behavioural one in an A/B table.
    """

    explorer = _bare_explorer()
    explorer.hazard_move_pruner = _AlwaysPrune()

    rows = explorer._candidates(_frame([1, 2, 3, 4]), path=[])

    assert len(rows) == 1, "a node must never be left with zero candidate actions"
    assert explorer._hazard_all_pruned_nodes == 1, "the guard event must be counted"
    assert explorer._hazard_rows_pruned == 3, (
        "the row put back by the guard must NOT count as withheld from the search"
    )
    diag = explorer.hazard_move_pruner_diagnostics()
    assert diag["all_pruned_nodes"] == 1
    # The pruner's own counter still sees 4 lethal verdicts; the explorer's net count is 3. Both
    # are reported so the guard's cost is visible rather than hidden.
    assert diag["pruner_prune_calls_lethal"] == 4
    assert diag["rows_pruned"] == 3


def test_scenario_arc_wmte_5970_a_raising_pruner_is_non_fatal_and_counted() -> None:
    """A raising ``should_prune`` must not break candidate generation (matching every sibling
    optional hook's try/except discipline) AND must be counted, so a silently-broken lever cannot
    masquerade as a lever that fired and found nothing."""

    class _Raises:
        def should_prune(self, frame: Any, label: Any) -> bool:
            raise RuntimeError("boom")

        def stats(self) -> dict:
            return {}

    explorer = _bare_explorer()
    explorer.hazard_move_pruner = _Raises()

    rows = explorer._candidates(_frame([1, 2]), path=[])

    assert {int(r["action"]) for r in rows} == {1, 2}, "candidates must pass through unchanged"
    assert explorer._hazard_prune_errors == 2, "every failure must be counted, not swallowed"
    assert explorer._hazard_rows_pruned == 0


def test_scenario_arc_wmte_5970_pruning_does_not_change_surviving_rows_tiers() -> None:
    """TIER INVARIANCE: the filter runs BEFORE the frontier-tier stamp, and ``row_tier`` is a pure
    function of a row, so dropping rows must not change any surviving row's tier. If it did, the
    pruner arm would be silently measuring a frontier-barrier delta as well."""

    from carnot.agentic.arc_frontier_discipline import row_tier

    # An empty tier map is the honest fixture here: `row_tier` is keyed on click COORDINATES, and
    # this frame's vocabulary is nav-only, so every row is tier 0 either way. What is being asserted
    # is that the mapping row -> tier is unchanged by dropping a sibling row.
    tier_by_xy: dict[tuple[int, int], int] = {}
    frame = _frame([1, 2, 3, 4])
    control = _bare_explorer()
    treated = _bare_explorer()
    treated.hazard_move_pruner = _PruneOneAction(2)

    control_rows = {
        int(r["action"]): row_tier(r, tier_by_xy) for r in control._candidates(frame, path=[])
    }
    treated_rows = {
        int(r["action"]): row_tier(r, tier_by_xy) for r in treated._candidates(frame, path=[])
    }

    assert set(treated_rows) < set(control_rows), "the treated arm must have dropped a row"
    for action, tier in treated_rows.items():
        assert tier == control_rows[action], f"action {action} changed tier under pruning"


# ============================================================================================
# 4. THE LIVE OBSERVE HOOK IN StepwiseExplorer._ingest -- the dead-channel class
# ============================================================================================


class _FakeFrame:
    """Minimal stand-in for an arcengine frame: only .frame is read by grid_of."""

    def __init__(self, grid: np.ndarray) -> None:
        self.frame = grid
        self.state = "NOT_FINISHED"
        self.levels_completed = 0


class _SpyObserve:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any, Any, bool]] = []

    def observe(
        self, frame_before: Any, label: Any, frame_after: Any, leveled_up: bool = False
    ) -> None:
        self.calls.append((frame_before, label, frame_after, leveled_up))

    def should_prune(self, frame: Any, label: Any) -> bool:
        return False

    def stats(self) -> dict:
        return {"observed": len(self.calls), "pruned": 0, "model_fitted": False}


def _drive_one_transition(explorer: StepwiseExplorer, action: int = 1) -> np.ndarray:
    """Ingest a root frame, reproduce ``_serve``'s ``awaiting`` construction FAITHFULLY (both frame
    fields resolved from the graph, so they are None on a bare explorer -- the whole point), then
    ingest the successor frame. Returns the antecedent grid."""

    grid0 = np.zeros((3, 3), dtype=int)
    explorer._ingest(_FakeFrame(grid0.copy()))
    origin = explorer.cur
    explorer.awaiting = {
        "origin": origin,
        "action": int(action),
        "data": None,
        "grid": explorer._grid_for_hash(origin),
        "level_before": int(explorer.best_level),
        "previous_frame": explorer.graph.get(origin, {}).get("frame"),
    }
    grid1 = grid0.copy()
    grid1[1, 1] = 5
    explorer._ingest(_FakeFrame(grid1))
    return grid0


def test_scenario_arc_wmte_5970_ingest_feeds_the_pruner_the_realized_transition() -> None:
    """SCENARIO-ARC-WMTE-5970-LIVE-WIRING-OBSERVE: the per-transition OBSERVE site feeds
    ``hazard_move_pruner.observe`` the realized (before, label, after, leveled_up) transition --
    the pruner's ONLY learning channel, since the hazard model is fit exclusively from deaths."""

    spy = _SpyObserve()
    explorer = StepwiseExplorer()
    explorer.hazard_move_pruner = spy

    grid0 = _drive_one_transition(explorer, action=1)

    assert len(spy.calls) == 1
    frame_before, label, frame_after, leveled_up = spy.calls[0]
    assert np.array_equal(np.asarray(getattr(frame_before, "frame", frame_before)), grid0)
    assert label == {"action": 1, "data": None}
    assert np.array_equal(frame_after.frame[1, 1], np.asarray(5))
    assert leveled_up is False


def test_scenario_arc_wmte_5970_observe_channel_survives_a_bare_explorer() -> None:
    """THE REGRESSION THAT MATTERS MOST (SCENARIO-ARC-WMTE-5970-FIRE-COUNTERS).

    The observe channel must NOT depend on any OTHER component. ``awaiting["previous_frame"]`` is
    ``graph[origin]["frame"]``, retained only when one of nine unrelated optional components is
    attached, and ``awaiting["grid"]`` reads the SAME field -- so an ``or o.get("grid")`` fallback
    rescues nothing. ``hazard_move_pruner`` is not among those nine. Measured consequence of getting
    this wrong: on the exp5836 ``CarnotAgentPolicy`` harness 0 of 122 graph nodes carried
    ``previous_frame`` and the sibling click pruner reported ``observed=0 pruned=0`` -- a
    byte-identical, zero-error NULL that was pure harness artifact, while on the scored
    ``E3AgentPolicy`` path 220 of 221 nodes carried one.

    This test does NOT hand-inject ``previous_frame`` (the defect an existing sibling test cannot
    catch for exactly that reason). It asserts the None precondition explicitly, so if node
    retention ever changes this test stops claiming to cover the bare case instead of silently
    passing for a new reason.
    """

    spy = _SpyObserve()
    explorer = _bare_explorer()
    explorer.hazard_move_pruner = spy

    grid0 = np.zeros((3, 3), dtype=int)
    explorer._ingest(_FakeFrame(grid0.copy()))
    origin = explorer.cur
    assert explorer.graph[origin].get("frame") is None, "precondition: no node frame retained"
    assert explorer._grid_for_hash(origin) is None, "precondition: the 'grid' fallback is dead too"

    explorer.awaiting = {
        "origin": origin,
        "action": 1,
        "data": None,
        "grid": explorer._grid_for_hash(origin),
        "level_before": int(explorer.best_level),
        "previous_frame": explorer.graph.get(origin, {}).get("frame"),
    }
    grid1 = grid0.copy()
    grid1[1, 1] = 5
    explorer._ingest(_FakeFrame(grid1))

    assert len(spy.calls) == 1, "OBSERVE CHANNEL IS DEAD on a bare explorer"
    frame_before = spy.calls[0][0]
    assert np.array_equal(np.asarray(frame_before), grid0), (
        "the antecedent must be the RAW GRID of the previous _ingest (self._last_grid), which the "
        "explorer maintains unconditionally -- not a retained node frame"
    )
    diag = explorer.hazard_move_pruner_diagnostics()
    assert diag["observe_calls"] == 1
    assert diag["antecedent_from_last_grid"] == 1, (
        "the witness that the component-independent fallback is what carried this transition"
    )
    assert diag["observe_errors"] == 0


def test_scenario_arc_wmte_5970_ingest_is_a_noop_when_the_lever_is_off() -> None:
    """No pruner configured -> ``_ingest`` runs cleanly and every counter stays zero."""

    explorer = _bare_explorer()
    assert explorer.hazard_move_pruner is None
    _drive_one_transition(explorer)
    diag = explorer.hazard_move_pruner_diagnostics()
    assert diag["enabled"] is False
    assert diag["observe_calls"] == 0
    assert diag["antecedent_from_last_grid"] == 0
    assert diag["rows_pruned"] == 0


def test_scenario_arc_wmte_5970_a_raising_observe_is_non_fatal_and_counted() -> None:
    class _Raises:
        def observe(self, *_a: Any, **_k: Any) -> None:
            raise RuntimeError("boom")

        def should_prune(self, frame: Any, label: Any) -> bool:
            return False

        def stats(self) -> dict:
            return {}

    explorer = _bare_explorer()
    explorer.hazard_move_pruner = _Raises()
    _drive_one_transition(explorer)
    assert explorer.hazard_move_pruner_diagnostics()["observe_errors"] == 1


def test_scenario_arc_wmte_5970_real_pruner_accumulates_over_a_driven_bare_run() -> None:
    """FIRE-COUNTER END-TO-END: the REAL pruner (not a spy) accumulates nav transitions through the
    live ``_ingest`` hook on a bare explorer. This is the property that makes a corpus null
    interpretable -- ``observed_nav_transitions > 0`` is what distinguishes "the lever fired and
    found nothing lethal" from "the lever never ran"."""

    explorer = _bare_explorer(hazard_move_pruner=True)
    assert isinstance(explorer.hazard_move_pruner, HazardMovePruner)

    grid = np.zeros((4, 4), dtype=int)
    explorer._ingest(_FakeFrame(grid.copy()))
    for step in range(1, 6):
        origin = explorer.cur
        explorer.awaiting = {
            "origin": origin,
            "action": 1 + (step % 4),
            "data": None,
            "grid": explorer._grid_for_hash(origin),
            "level_before": int(explorer.best_level),
            "previous_frame": explorer.graph.get(origin, {}).get("frame"),
        }
        nxt = grid.copy()
        nxt[step % 4, step % 4] = step
        explorer._ingest(_FakeFrame(nxt))
        grid = nxt

    diag = explorer.hazard_move_pruner_diagnostics()
    assert diag["observe_calls"] == 5, "every transition must reach the hook"
    assert diag["observed_nav_transitions"] == 5, (
        "and the pruner must ACCEPT them -- a nonzero observe_calls with zero accepted "
        "transitions would mean the pruner is silently dropping the live label shape"
    )
    assert diag["antecedent_from_last_grid"] == 5, "carried entirely by the unconditional fallback"
    assert diag["observe_errors"] == 0
    # No deaths were staged, so the honest outcome is UNFITTED -- and the counters say so, which is
    # the whole point: this is a legible non-firing, not a null.
    assert diag["model_fitted"] is False
    assert diag["n_deaths"] == 0
    assert diag["rows_pruned"] == 0


# ============================================================================================
# 5. DEFAULT-OFF PARITY AND THE FLAG LADDER
# ============================================================================================


def test_scenario_arc_wmte_5970_default_off_everywhere() -> None:
    """SCENARIO-ARC-WMTE-5970-DEFAULT-OFF-PARITY: the module flag, the explorer, and both policy
    classes all default the lever OFF, so no scored-path behaviour moves until an operator flips
    it. New behaviour ships default-off -- this test is what makes that mechanical."""

    assert SUBMITTED_HAZARD_MOVE_PRUNER_ENABLED is False
    assert SUBMITTED_AGENT_CONFIG["hazard_move_pruner_enabled"] is False
    assert SUBMITTED_AGENT_CONFIG["hazard_move_pruner_wired"] is True, (
        "the config must still record that the lever EXISTS on the live path -- an unwired "
        "lever and a wired-but-off lever are different states"
    )

    assert StepwiseExplorer().hazard_move_pruner is None
    assert E3AgentPolicy("tu93").explorer.hazard_move_pruner is None
    assert CarnotAgentPolicy("tu93", {}, force_explore=True).explorer.hazard_move_pruner is None


def test_scenario_arc_wmte_5970_default_off_leaves_candidate_generation_byte_identical() -> None:
    """With the lever off, candidate generation must be IDENTICAL to the pre-change behaviour.

    Mechanically, "off" means ``self.hazard_move_pruner is None`` and both hooks are guarded by
    ``is not None``, so no hazard code executes at all. This test pins the observable consequence:
    the default explorer and one with the lever explicitly disabled produce identical rows in
    identical order, and no counter moves.
    """

    frame = _frame([1, 2, 3, 4, 6])
    default_rows = _bare_explorer()._candidates(frame, path=[])
    explicit_off = _bare_explorer(hazard_move_pruner=False)
    off_rows = explicit_off._candidates(frame, path=[])

    def _key(rows: list[dict]) -> list[tuple]:
        return [
            (int(r["action"]), (r.get("data") or {}).get("x"), (r.get("data") or {}).get("y"))
            for r in rows
        ]

    assert _key(default_rows) == _key(off_rows), "order and content must both be unchanged"
    assert explicit_off._hazard_rows_pruned == 0
    assert explicit_off._hazard_all_pruned_nodes == 0
    assert explicit_off._hazard_prune_errors == 0


def test_scenario_arc_wmte_5970_flag_ladder_is_kwarg_then_env_then_default(monkeypatch) -> None:
    """The gated-flag resolution order is explicit-kwarg > env override > ``SUBMITTED_*`` default,
    matching every sibling gated flag. The env path is what lets an A/B harness flip ONE arm
    without mutating module globals, which would leak across arms inside a single process."""

    monkeypatch.setenv("CARNOT_ARC_HAZARD_MOVE_PRUNER", "1")
    assert StepwiseExplorer().hazard_move_pruner is not None, "env override must enable"
    assert StepwiseExplorer(hazard_move_pruner=False).hazard_move_pruner is None, (
        "an explicit kwarg must OUTRANK the env override"
    )

    monkeypatch.setenv("CARNOT_ARC_HAZARD_MOVE_PRUNER", "0")
    assert StepwiseExplorer().hazard_move_pruner is None
    assert StepwiseExplorer(hazard_move_pruner=True).hazard_move_pruner is not None

    monkeypatch.delenv("CARNOT_ARC_HAZARD_MOVE_PRUNER")
    assert StepwiseExplorer().hazard_move_pruner is None, "falls back to the SUBMITTED_* default"


def test_scenario_arc_wmte_5970_a_prebuilt_instance_counts_as_an_explicit_enable() -> None:
    """An injected pruner instance must be USED, not silently discarded by the boolean gate --
    otherwise a test or arm that widens ``refit_every`` would get a default instance instead and
    measure the wrong thing."""

    injected = HazardMovePruner(np.asarray, refit_every=7, min_deaths=1)
    explorer = StepwiseExplorer(hazard_move_pruner=injected)
    assert explorer.hazard_move_pruner is injected
    assert explorer.hazard_move_pruner_enabled is True
    assert explorer.hazard_move_pruner_diagnostics()["min_deaths"] == 1


def test_scenario_arc_wmte_5970_policies_pass_the_flag_through_to_the_explorer() -> None:
    """Both policy classes must actually forward the kwarg. Without this the flag would appear
    settable while the scored explorer kept its default -- a lever that looks wired and is not."""

    assert E3AgentPolicy("tu93", hazard_move_pruner=True).explorer.hazard_move_pruner is not None
    policy = CarnotAgentPolicy("tu93", {}, force_explore=True, hazard_move_pruner=True)
    assert policy.explorer.hazard_move_pruner is not None


def test_scenario_arc_wmte_5970_module_is_in_the_scored_import_closure() -> None:
    """The module must be imported at TOP LEVEL of the scored agent, so
    ``scripts/arc_orphan_solver_lint.py`` sees it in the live closure. Before this change
    ``arc_hazard_pruner`` was reachable only from the offline dev twin -- which is itself an allowed
    live entrypoint, so the lint PASSED while the pruner had never touched the scored path."""

    import carnot.agentic.arc_competition_agent as comp

    source = open(comp.__file__, encoding="utf-8").read()
    assert "from carnot.agentic.arc_hazard_pruner import coerce_hazard_move_pruner" in source, (
        "the import must be module-scope, not inside a function -- the lint parses top-level "
        "imports and a function-local import would leave the module orphaned"
    )
