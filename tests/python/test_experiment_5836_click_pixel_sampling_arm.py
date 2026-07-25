"""REQ-ARC-WMTE-5950 harness wiring: the sampler arms, the pre-registered gate, and the
arm-E positive-control repairs (seeding + instrumentation).

Every test asserts and none is skipped: all of this is pure analysis code over row dicts,
plus one seeding-order test that uses a fake agent factory rather than the vendored
reference clone (so it runs whether or not that clone is present).
"""

from __future__ import annotations

import random
import sys
import types

import pytest

from carnot import experiment_5836_frontier_discipline_ab as ab

# Importing the harness pulls in arc_competition_agent, whose own import chain loads torch /
# jax (~650MB RSS). That is a one-off IMPORT footprint, not a per-test leak: it does not
# appear when the file runs inside the wider suite (where something else has already imported
# them), only when this file runs in isolation and the first test in each worker is charged for
# it. Same marker, same reason, as test_arc_goal_predicate_live_veto.py and
# test_arc_relational_goal_energy_live_5711.py. The marker suppresses the RSS TEARDOWN check
# only -- every test below still runs and still asserts.
pytestmark = pytest.mark.memory_watchdog_skip


# ---------------------------------------------------------------------------
# The arms themselves
# ---------------------------------------------------------------------------


def test_sampler_arms_differ_from_their_matched_control_by_exactly_one_mechanism():
    """F/F1 minus B2 must be ONLY the sampler, or the A/B cannot attribute a delta."""

    control = ab.ARMS[ab.CLICK_PIXEL_CONTROL_ARM]["kwargs"]
    for arm in ab.CLICK_PIXEL_ARMS:
        kw = dict(ab.ARMS[arm]["kwargs"])
        assert kw.pop("click_pixel_sampling") is True
        kw.pop("click_pixel_redraw_budget", None)
        assert kw == control, arm
        assert ab.ARMS[arm]["deterministic"] is False  # it draws -> it needs seeds


def test_arm_a_is_the_pre_flip_baseline_not_the_live_configuration():
    """The correction that makes B2 the right control.

    Arm A passes tier_exhaustion=False as an explicit constructor kwarg, and the explorer's
    _fd_gate ranks an explicit kwarg above the SUBMITTED_* default -- so arm A did NOT
    inherit the 2026-07-25 flip and is not today's agent.
    """

    from carnot.agentic import arc_competition_agent as aca

    assert ab.ARMS["A"]["kwargs"]["tier_exhaustion"] is False
    assert aca.SUBMITTED_FRONTIER_TIER_EXHAUSTION_ENABLED is True  # the live default
    assert ab.ARMS["B2"]["kwargs"]["tier_exhaustion"] is True
    assert ab.ARMS["B2"]["kwargs"]["tier_uniform_random"] is True
    assert aca.SUBMITTED_FRONTIER_TIER_UNIFORM_RANDOM_ENABLED is True
    assert ab.CLICK_PIXEL_CONTROL_ARM == "B2"


def test_grafted_arms_definition_was_not_silently_redefined():
    """Adding arms must not change an already-published quantity's definition."""

    assert ab.GRAFTED_ARMS == ("B", "B2", "C", "D")
    assert set(ab.CLICK_PIXEL_ARMS) & set(ab.GRAFTED_ARMS) == set()


def test_sampler_arms_get_multiple_seeds():
    assert ab._seeds_for("A", 3) == [ab.RANDOM_SEED]  # deterministic -> one run
    for arm in ab.CLICK_PIXEL_ARMS:
        assert len(ab._seeds_for(arm, 3)) == 3


# ---------------------------------------------------------------------------
# The pre-registered gate
# ---------------------------------------------------------------------------


def _row(arm, game, seed, levels, condition="real"):
    return {
        "arm": arm,
        "game": game,
        "seed": seed,
        "condition": condition,
        "ran": True,
        "levels": levels,
    }


def test_gate_pass_region_is_provably_non_empty():
    """The exp5835 defect guard: a gate whose conjunction can never be satisfied.

    The witness is COMPUTED with the same predicate the gate uses, so tightening the gate
    into an unsatisfiable shape turns this False instead of silently reporting a null.
    """

    witness = ab._click_pixel_gate_pass_region_witness()
    assert witness["passes"] is True
    assert witness["gained"] is True and witness["regressed"] is False
    gate = ab.click_pixel_sampling_gate([])
    assert gate["pass_region_nonempty"] is True


def test_gate_passes_on_a_real_new_win_without_a_regression():
    rows = [
        _row("B2", "gA", 1, 1),
        _row("B2", "gB", 1, 0),
        _row("F", "gA", 1, 1),
        _row("F", "gB", 1, 1),  # a NEW win on the same seed
    ]
    gate = ab.click_pixel_sampling_gate(rows)
    assert gate["passed"] is True
    assert gate["per_arm"]["F"]["per_seed"][0]["new_wins"] == ["gB"]
    assert gate["per_arm"]["F"]["any_seed_regressed"] is False


def test_gate_fails_when_a_win_is_given_back_on_any_seed():
    """The any-seed-UNION failure this project already made once.

    A union over seeds says "F won gA and gB" and passes. Per seed, F LOST gA on seed 2, so
    the gate must fail.
    """

    rows = [
        _row("B2", "gA", 1, 1),
        _row("B2", "gA", 2, 1),
        _row("B2", "gB", 1, 0),
        _row("B2", "gB", 2, 0),
        _row("F", "gA", 1, 1),
        _row("F", "gA", 2, 0),  # regression on seed 2
        _row("F", "gB", 1, 1),  # gain on seed 1
        _row("F", "gB", 2, 0),
    ]
    gate = ab.click_pixel_sampling_gate(rows)
    assert gate["per_arm"]["F"]["any_seed_gained"] is True
    assert gate["per_arm"]["F"]["any_seed_regressed"] is True
    assert gate["passed"] is False
    # and the union really would have passed -- which is why the union is not used
    union_new = {"gA", "gB"} - {"gA"}
    assert union_new == {"gB"}


def test_gate_fails_loudly_when_the_matched_control_was_not_measured():
    gate = ab.click_pixel_sampling_gate([_row("F", "gA", 1, 1)])
    assert gate["passed"] is False
    assert "matched control" in gate["reason"]


def test_gate_reports_an_arm_with_no_shared_seed_as_unmeasured_not_as_a_pass():
    rows = [_row("B2", "gA", 1, 1), _row("F", "gA", 99, 1), _row("F", "gB", 99, 1)]
    gate = ab.click_pixel_sampling_gate(rows)
    assert gate["per_arm"]["F"]["measured"] is False
    assert gate["passed"] is False


def test_gate_is_flat_mirrored_and_kept_out_of_the_pre_existing_all_passed_field():
    cap = {"available": False}
    rows = [_row("B2", "gA", 1, 1), _row("F", "gA", 1, 1), _row("F", "gB", 1, 1)]
    gates = ab.acceptance_gates(cap, {}, {}, rows)
    assert gates["acceptance_gate_click_pixel_sampling_passed"] is True
    # the pre-existing verdict field must NOT be moved by the new mechanism
    assert gates["acceptance_gates_all_passed"] is False
    # and omitting rows keeps the artifact shape backward compatible
    assert "acceptance_gate_click_pixel_sampling" not in ab.acceptance_gates(cap, {}, {})


def test_per_seed_win_sets_records_a_measured_zero_rather_than_omitting_the_seed():
    """A seed that won nothing must appear as an EMPTY set, not be missing.

    If it were missing, `set(treat) & set(control)` would drop it from the shared-seed list
    and a total failure on that seed would be invisible.
    """

    sets = ab._per_seed_win_sets([_row("F", "gA", 5, 0)], "F")
    assert sets == {5: set()}


# ---------------------------------------------------------------------------
# Arm E: the positive-control repairs
# ---------------------------------------------------------------------------


class _SeedClobberingAgent:
    """Mimics HeuristicAgent.__init__, which reseeds the GLOBAL RNG from the wall clock."""

    def __init__(self) -> None:
        import time

        random.seed(int(time.time() * 1000000) + 12345)


def test_reference_agent_construction_restores_the_harness_seed():
    """THE arm-E seeding fix, asserted on order rather than on text.

    Constructing the reference clobbers the seed; the harness must restore it AFTER
    construction, otherwise every arm-E cell is an unrepeatable draw (the defect that forced
    the 2026-07-24 corrigendum).
    """

    agent = ab.construct_reference_agent_seeded(_SeedClobberingAgent, 777)
    assert isinstance(agent, _SeedClobberingAgent)
    got = [random.random() for _ in range(5)]
    expected = random.Random(777).random(), *[None] * 0
    assert got[0] == expected[0]

    # And the whole sequence is reproducible: two constructions with the same seed produce
    # the same draws, which is what makes an arm-E cell re-runnable.
    ab.construct_reference_agent_seeded(_SeedClobberingAgent, 777)
    again = [random.random() for _ in range(5)]
    assert got == again

    # Different seed -> different stream (so the seeds are a real replication axis).
    ab.construct_reference_agent_seeded(_SeedClobberingAgent, 778)
    other = [random.random() for _ in range(5)]
    assert other != got


def test_seed_global_rngs_seeds_numpy_too():
    import numpy as np

    ab._seed_global_rngs(31)
    a = np.random.rand(4).tolist()
    ab._seed_global_rngs(31)
    assert np.random.rand(4).tolist() == a


# ---------------------------------------------------------------------------
# Every arm emits an expansion count AND an error count
# ---------------------------------------------------------------------------


def test_run_cell_attributes_a_crash_to_the_arm_that_crashed(monkeypatch):
    """The uninstrumented-arm failure: a crashed arm must not read as a clean null."""

    class _Explorer:
        graph = {"a": {}, "b": {}}

    class _Policy:
        explorer = _Explorer()

    stub = types.ModuleType("arc_leaderboard_eval")

    def _boom(*_a, **_k):
        raise RuntimeError("env exploded")

    stub.run_game = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "arc_leaderboard_eval", stub)
    monkeypatch.setattr(ab, "_explorer_policy", lambda game, **kw: _Policy())

    row = ab.run_cell("F", "gA", budget=10, seed=1, variant=0, reflect=None)
    assert row["ran"] is False
    assert row["errors"] == 1
    assert row["states_expanded"] == 2  # effort still reported, not None
    assert "RuntimeError" in row["reason"]


def test_run_cell_emits_zero_errors_on_a_clean_run(monkeypatch):
    class _Explorer:
        graph = {"a": {}}
        target_levels = 1

        def frontier_discipline_diagnostics(self):
            return {"click_pixel_errors": 0, "click_pixel_sampling_enabled": True}

    class _Policy:
        explorer = _Explorer()

    stub = types.ModuleType("arc_leaderboard_eval")
    stub.run_game = lambda *a, **k: {  # type: ignore[attr-defined]
        "levels": 1,
        "reached": 1,
        "actions": 12,
        "actions_to_first_levelup": 12,
        "efficiency": 1.5,
    }
    monkeypatch.setitem(sys.modules, "arc_leaderboard_eval", stub)
    monkeypatch.setattr(ab, "_explorer_policy", lambda game, **kw: _Policy())

    row = ab.run_cell("F", "gA", budget=10, seed=1, variant=0, reflect=None)
    assert row["ran"] is True
    assert row["errors"] == 0
    assert row["states_expanded"] == 1
    assert row["frontier_discipline"]["click_pixel_sampling_enabled"] is True


def test_arm_e_row_carries_states_expanded_and_errors(monkeypatch):
    """Arm E used to hardcode states_expanded=None on every row."""

    def _runner(game, *, budget, seed, variant, reflect):
        return {
            "reached": 1,
            "levels": 1,
            "actions": 700,
            "actions_to_first_levelup": 20,
            "actions_to_first_levelup_incl_reset": 21,
            "resets_taken": 1,
            "states_expanded": 44,
            "errors": 3,
            "error_breakdown": {"take_action_swallowed": 3, "choose_action_raised": 0},
            "duration_s": 1.0,
        }

    row = ab.run_cell("E", "gA", budget=700, seed=1, variant=0, reflect=None, je_runner=_runner)
    assert row["states_expanded"] == 44
    assert row["errors"] == 3
    assert row["error_breakdown"]["take_action_swallowed"] == 3


def test_arm_e_without_a_runner_is_recorded_as_not_run():
    row = ab.run_cell("E", "gA", budget=10, seed=1, variant=0, reflect=None, je_runner=None)
    assert row["ran"] is False
    assert row["reason"] == "no_reference_runner"


def test_load_just_explore_runner_returns_a_reason_string_either_way():
    runner, reason = ab.load_just_explore_runner()
    assert isinstance(reason, str) and reason
    if runner is None:
        assert reason.startswith(("reference_clone_absent", "shim_absent", "shim_import_failed"))
    else:
        assert reason == "ok"
        assert callable(runner)


def test_conditions_and_arms_stay_consistent():
    with pytest.raises(KeyError):
        ab.ARMS["not_an_arm"]
    assert set(ab.CLICK_PIXEL_ARMS) <= set(ab.ARMS)
    assert ab.CLICK_PIXEL_CONTROL_ARM in ab.ARMS


# ---------------------------------------------------------------------------
# 2026-07-25 adversarial-review repairs
# ---------------------------------------------------------------------------


def _win_row(arm, game, seed, **extra):
    row = {
        "arm": arm,
        "game": game,
        "seed": seed,
        "condition": "real",
        "ran": True,
        "levels": 1,
        "actions": 100,
    }
    row.update(extra)
    return row


def _loss_row(arm, game, seed, **extra):
    row = _win_row(arm, game, seed, **extra)
    row["levels"] = 0
    return row


def test_reproduction_gate_never_drops_a_claim_carrying_arm_to_the_limit():
    """5 arms with wins + limit=4 previously dropped F1 -- one of the two claim arms.

    The round-robin was correct; the LIMIT truncated it. The effective limit is now floored at
    the number of arms with wins, so no winning arm can be silently excluded from the sample.
    """

    rows = [
        _win_row(arm, "gA", 1)
        for arm in ("A", ab.CLICK_PIXEL_CONTROL_ARM, "E") + tuple(ab.CLICK_PIXEL_ARMS)
    ]
    calls: list[tuple] = []

    def _fake_cell(arm, game, **kwargs):
        calls.append((arm, game))
        return {"levels": 1}

    # replay_validate calls run_cell as a module global, so swapping it here exercises the real
    # selection logic without launching 5 offline games.
    original = ab.run_cell
    try:
        ab.run_cell = _fake_cell  # type: ignore[assignment]
        out = ab.replay_validate(rows, budget=10, limit=4)
    finally:
        ab.run_cell = original  # type: ignore[assignment]

    assert out["replay_limit_requested"] == 4
    assert out["replay_limit_effective"] == 5  # floored at the 5 arms with wins
    assert out["n_arms_with_wins"] == 5
    assert out["arms_not_reproduced"] == []
    assert out["claim_carrying_arms_not_reproduced"] == []
    for arm in ab.CLICK_PIXEL_ARMS:
        assert arm in out["arms_reproduced"], f"{arm} carries the claim and must be checked"
    assert out["all_arms_with_wins_reproduced"] is True


def test_reproduction_gate_names_a_dropped_arm_rather_than_reporting_a_clean_pass():
    """Sanity: the new field is a real check, not a constant. A row that cannot be replayed
    (unknown condition) is skipped by the loop, and the arm must then be NAMED as unreproduced
    instead of the gate reporting n/n."""

    rows = [_win_row("A", "gA", 1), _win_row("F", "gA", 1)]
    rows[1]["condition"] = "not_a_declared_condition"

    def _fake_cell(arm, game, **kwargs):
        return {"levels": 1}

    original = ab.run_cell
    try:
        ab.run_cell = _fake_cell  # type: ignore[assignment]
        out = ab.replay_validate(rows, budget=10, limit=6)
    finally:
        ab.run_cell = original  # type: ignore[assignment]

    assert "F" in out["arms_not_reproduced"]
    assert out["claim_carrying_arms_not_reproduced"] == ["F"]
    assert out["all_arms_with_wins_reproduced"] is False


def test_gate_reports_uninformative_when_the_control_wins_every_measured_game():
    """The smoke's real shape: control at ceiling on every game -> the pass region is EMPTY.

    ``pass_region_witness`` cannot detect this (it is synthetic), so the gate must compute
    reachable headroom from the measured rows and refuse to call the result a null.
    """

    control = ab.CLICK_PIXEL_CONTROL_ARM
    rows = []
    for seed in (1, 2):
        for game in ("gA", "gB"):
            rows.append(_win_row(control, game, seed))
            for arm in ab.CLICK_PIXEL_ARMS:
                rows.append(_win_row(arm, game, seed))
    gate = ab.click_pixel_sampling_gate(rows)
    assert gate["passed"] is False
    assert gate["headroom_present"] is False
    assert gate["informative"] is False
    assert gate["verdict"] == "uninformative_no_headroom"
    assert gate["reachable_new_win_games"] == []
    for arm in ab.CLICK_PIXEL_ARMS:
        assert gate["per_arm"][arm]["headroom_present"] is False
        assert gate["per_arm"][arm]["per_seed"][0]["max_attainable_new_wins"] == 0


def test_gate_is_informative_when_the_control_loses_a_game_both_arms_measured():
    control = ab.CLICK_PIXEL_CONTROL_ARM
    rows = []
    for seed in (1, 2):
        rows.append(_win_row(control, "gA", seed))
        rows.append(_loss_row(control, "gB", seed))
        for arm in ab.CLICK_PIXEL_ARMS:
            rows.append(_win_row(arm, "gA", seed))
            rows.append(_loss_row(arm, "gB", seed))
    gate = ab.click_pixel_sampling_gate(rows)
    assert gate["headroom_present"] is True
    assert gate["informative"] is True
    assert gate["reachable_new_win_games"] == ["gB"]
    assert gate["verdict"] == "failed_with_headroom_present"
    assert gate["passed"] is False  # a real, informative null


def test_gate_reports_mechanism_activity_per_arm():
    control = ab.CLICK_PIXEL_CONTROL_ARM
    rows = [_win_row(control, "gA", 1), _loss_row(control, "gB", 1)]
    rows.append(_win_row("F", "gA", 1, click_pixel_coordinates_changed=41))
    rows.append(_loss_row("F", "gB", 1, click_pixel_coordinates_changed=17))
    rows.append(_win_row("F1", "gA", 1, click_pixel_coordinates_changed=0))
    rows.append(_loss_row("F1", "gB", 1, click_pixel_coordinates_changed=0))
    gate = ab.click_pixel_sampling_gate(rows)
    assert gate["per_arm"]["F"]["mechanism_active"]["active"] is True
    assert gate["per_arm"]["F"]["mechanism_active"]["coordinates_changed"] == 58
    # a treatment arm that replaced nothing is a CONTROL, and the gate says so
    assert gate["per_arm"]["F1"]["mechanism_active"]["active"] is False
    assert gate["per_arm"]["F1"]["mechanism_active"]["instrumented"] is True


def test_uninstrumented_rows_are_reported_as_uninstrumented_not_as_inactive():
    control = ab.CLICK_PIXEL_CONTROL_ARM
    rows = [_win_row(control, "gA", 1), _win_row("F", "gA", 1)]  # no witness field at all
    gate = ab.click_pixel_sampling_gate(rows)
    assert gate["per_arm"]["F"]["mechanism_active"]["instrumented"] is False
    assert gate["per_arm"]["F"]["mechanism_active"]["active"] is False


def test_a_degenerate_positive_control_is_not_a_usable_positive_control():
    """The measured smoke shape: 4 of 6 arm-E cells at 78-96% choose_action-raised.

    ``errored_cell_rate`` sees none of this (every cell ``ran``), so the health check has to
    read the within-cell fallback fraction.
    """

    rows = []
    for game, err in (("gA", 1610), ("gB", 1701), ("gC", 1927), ("gD", 1776)):
        rows.append(
            _win_row(
                "E",
                game,
                1,
                actions=2001,
                errors=err,
                reference_choose_action_raised=err,
                degenerate_fallback_fraction=round(err / 2001, 4),
            )
        )
    rows.append(_win_row("E", "gE", 1, actions=2001, errors=0, degenerate_fallback_fraction=0.0))
    health = ab.positive_control_health(rows)
    assert health["measured"] is True
    assert health["n_cells"] == 5
    assert health["n_degenerate_cells"] == 4
    assert health["healthy"] is False
    assert health["reason"].startswith("reference_degenerate_in_4_of_5_cells")
    assert health["worst_cell_fallback_fraction"] > 0.9


def test_a_clean_positive_control_is_healthy():
    rows = [
        _win_row("E", g, 1, actions=2001, errors=0, degenerate_fallback_fraction=0.0)
        for g in ("gA", "gB")
    ]
    health = ab.positive_control_health(rows)
    assert health["healthy"] is True
    assert health["reason"] == "ok"
    assert health["n_degenerate_cells"] == 0


def test_degenerate_control_flips_the_verdict_and_the_diagnostic_target():
    unhealthy = {
        "healthy": False,
        "n_degenerate_cells": 4,
        "n_cells": 6,
        "worst_cell_fallback_fraction": 0.96,
    }
    verdict = ab.verdict_for(
        {"full_declared_spec": True},
        {"available": True, "new_wins_vs_baseline": 0, "positive_control_new_wins": 1},
        positive_control_ran=True,
        error_rate=0.0,
        control_health=unhealthy,
    )
    assert "uninterpretable_arm_error_rate" in verdict
    assert verdict.startswith("complete_")  # terminal prefix preserved

    cap = ab.capability_summary(
        {"E|real": {"games_won_any_seed": ["gZ"]}},
        {
            "available": True,
            "baseline_wins": [],
            "E": {"new_wins": ["gZ"]},
        },
        control_healthy=False,
    )
    assert cap["positive_control_healthy"] is False
    assert "NOT YET ATTRIBUTABLE TO THE REFERENCE" in cap["diagnostic_target"]


def test_verdict_names_an_uninformative_sampler_gate():
    verdict = ab.verdict_for(
        {"full_declared_spec": False, "n_games": 3, "budget": 2000},
        {"available": True, "new_wins_vs_baseline": 0, "positive_control_new_wins": 1},
        positive_control_ran=True,
        error_rate=0.0,
        control_health={"healthy": True},
        cps_gate={"informative": False, "passed": False},
    )
    assert "sampler_untested_no_headroom" in verdict
    assert "sampler_gate_failed" not in verdict


def test_verdict_distinguishes_an_informative_sampler_null():
    verdict = ab.verdict_for(
        {"full_declared_spec": False, "n_games": 25, "budget": 2000},
        {"available": True, "new_wins_vs_baseline": 0, "positive_control_new_wins": 0},
        positive_control_ran=True,
        error_rate=0.0,
        control_health={"healthy": True},
        cps_gate={"informative": True, "passed": False},
    )
    assert "sampler_gate_failed_with_headroom" in verdict


def test_arm_e_row_carries_the_degeneracy_fields():
    def _runner(game, *, budget, seed, variant, reflect):
        return {
            "levels": 1,
            "reached": 1,
            "actions": 2000,
            "actions_to_first_levelup": 20,
            "states_expanded": 7,
            "errors": 1900,
            "error_breakdown": {"take_action_swallowed": 0, "choose_action_raised": 1900},
            "duration_s": 1.0,
        }

    row = ab.run_cell("E", "gA", budget=2000, seed=1, variant=0, reflect=None, je_runner=_runner)
    assert row["reference_choose_action_raised"] == 1900
    assert row["degenerate_fallback_fraction"] == 0.95
    assert row["reference_degenerate"] is True


def test_artifact_identity_constants_are_distinct_requirements():
    assert ab.EXPERIMENT_ID == 5836
    assert ab.CLICK_PIXEL_EXPERIMENT_ID == 5950
    assert ab.EXPERIMENT_ID != ab.CLICK_PIXEL_EXPERIMENT_ID


def test_headline_leads_with_the_sampler_on_a_sampler_run():
    cap = {"available": True, "new_wins_vs_baseline": 0, "positive_control_new_wins": 1}
    gate = {
        "informative": True,
        "passed": False,
        "headroom_narrow": True,
        "n_reachable_new_win_games": 1,
        "reachable_new_win_games": ["r11l"],
        "per_arm": {
            "F": {"mechanism_active": {"active": True, "coordinates_changed": 20007}},
            "F1": {"mechanism_active": {"active": True, "coordinates_changed": 11856}},
        },
    }
    line = ab.build_headline(
        cap,
        control_health={"healthy": True},
        cps_gate=gate,
        sampler_run=True,
    )
    assert line.startswith("REQ-ARC-WMTE-5950 click-pixel sampling")
    assert "31863 click coordinates replaced" in line
    assert "ONE-game win axis" in line
    assert "grafted tier exhaustion" in line  # the graft result is still reported, just not first


def test_headline_says_untested_not_null_when_the_gate_has_no_headroom():
    line = ab.build_headline(
        {"available": True, "new_wins_vs_baseline": 0, "positive_control_new_wins": 1},
        control_health={"healthy": True},
        cps_gate={"informative": False, "passed": False, "per_arm": {}},
        sampler_run=True,
    )
    assert "UNTESTED for capability" in line
    assert "NOT a measured null" in line


def test_headline_does_not_assert_identical_conditions_for_a_degenerate_control():
    healthy_line = ab.build_headline(
        {"available": True, "new_wins_vs_baseline": 0, "positive_control_new_wins": 1},
        control_health={"healthy": True},
    )
    assert "under identical conditions" in healthy_line

    degenerate_line = ab.build_headline(
        {"available": True, "new_wins_vs_baseline": 0, "positive_control_new_wins": 1},
        control_health={
            "healthy": False,
            "n_degenerate_cells": 5,
            "n_cells": 6,
            "worst_cell_fallback_fraction": 0.963,
        },
    )
    assert "DEGENERATE on 5 of 6 cells" in degenerate_line
    assert "does NOT hold" in degenerate_line


def test_narrow_headroom_is_flagged_distinctly_from_real_headroom():
    control = ab.CLICK_PIXEL_CONTROL_ARM
    rows = [_win_row(control, "gA", 1), _loss_row(control, "gB", 1)]
    for arm in ab.CLICK_PIXEL_ARMS:
        rows.append(_win_row(arm, "gA", 1))
        rows.append(_loss_row(arm, "gB", 1))
    gate = ab.click_pixel_sampling_gate(rows)
    assert gate["n_reachable_new_win_games"] == 1
    assert gate["headroom_narrow"] is True
    assert "ONE candidate game" in gate["headroom_narrow_note"]

    rows.append(_loss_row(control, "gC", 1))
    for arm in ab.CLICK_PIXEL_ARMS:
        rows.append(_loss_row(arm, "gC", 1))
    wider = ab.click_pixel_sampling_gate(rows)
    assert wider["n_reachable_new_win_games"] == 2
    assert wider["headroom_narrow"] is False
    assert "headroom_narrow_note" not in wider
