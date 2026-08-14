"""REQ-ARC-BENCH-6267 / REQ-ARC-FLAG-LEDGER-6268.

The ARC loop had generation and no selection: 101 `CARNOT_ARC_*` flags, 13 of 16 recent tasks
ending `ready_no_solve_claim` or `default_off`, and three "holdout" experiments whose metric sets
share zero keys. `arc_bench.py` produces one comparable number; `arc_flag_ledger.py` selects on it.

These tests cover the DECISION logic, not the arcade. Running a game needs the offline arcade and
takes seconds per game; the part that can silently be wrong is the rule that decides what turns
on, so that is what is pinned here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import arc_bench  # noqa: E402
import arc_flag_ledger as ledger  # noqa: E402


def _row(game, levels, actions, error=None):
    return {
        "game": game,
        "levels_cleared": levels,
        "actions_spent": actions,
        "solution_len": 0,
        "wall_s": 0.1,
        "error": error,
        "adapter_used": False,
    }


def _report(rows):
    return {"per_game_rows": rows, **arc_bench.aggregate(rows)}


# --------------------------------------------------------------------------- aggregate


def test_aggregate_reports_errors_instead_of_dropping_them():
    """A benchmark that silently drops its hard cases reports a rising average while getting worse."""
    agg = arc_bench.aggregate([_row("a", 1, 100), _row("b", 0, 500), _row("c", 0, 0, "boom")])

    assert agg["games_run"] == 3
    assert agg["games_errored"] == 1
    assert agg["total_levels_cleared"] == 1
    assert agg["clear_rate"] == 0.5, "the errored game is excluded from the rate, not counted clean"


def test_actions_per_level_is_none_not_a_sentinel_when_nothing_cleared():
    """ "Infinitely inefficient" and "did not run" are different facts.

    A sentinel like 0 or a huge float would be averaged into a later comparison as if it meant
    something.
    """
    agg = arc_bench.aggregate([_row("a", 0, 900), _row("b", 0, 100)])

    assert agg["total_levels_cleared"] == 0
    assert agg["actions_per_level_cleared"] is None


# --------------------------------------------------------------------------- rotation


def test_rotation_advances_so_successive_runs_cover_the_roster(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "l.yaml")
    monkeypatch.setattr(arc_bench, "ROTATION", tmp_path / "rot.json")
    games = [f"g{i}" for i in range(10)]

    first = arc_bench.select(games, 4)
    second = arc_bench.select(games, 4)

    assert first == ["g0", "g1", "g2", "g3"]
    assert second == ["g4", "g5", "g6", "g7"], "a fixed offset would re-measure the same head slice"


def test_rotation_resets_when_the_roster_changes(tmp_path, monkeypatch):
    """The QA-audit `units_signature` lesson, applied here.

    An offset persisted against a roster that has since changed silently skips the newly-added
    games -- sometimes for many runs. Re-measuring some games is cheap; a new game never being
    benchmarked is the failure that matters, so a changed roster resolves the offset to 0.
    """
    monkeypatch.setattr(arc_bench, "ROTATION", tmp_path / "rot.json")
    arc_bench.select([f"g{i}" for i in range(10)], 4)  # offset -> 4

    picked = arc_bench.select([f"h{i}" for i in range(10)], 4)  # different roster

    assert picked == ["h0", "h1", "h2", "h3"], "a changed roster must restart from the beginning"


def test_rotation_survives_a_corrupt_state_file(tmp_path, monkeypatch):
    rot = tmp_path / "rot.json"
    rot.write_text("{not json")
    monkeypatch.setattr(arc_bench, "ROTATION", rot)

    assert arc_bench.select([f"g{i}" for i in range(6)], 2) == ["g0", "g1"]


# --------------------------------------------------------------------------- the regression gate


def test_promotion_is_refused_when_any_single_game_regresses():
    """THE load-bearing test. An aggregate win that costs one game is the ka59 failure repeating.

    The ARC engine store overwrote unconditionally and destroyed ka59 from 1.0 to 0.0; retention
    was worth p=4.9e-5. Here the arm clears three NEW games and loses one. Total levels go UP. It
    must still be refused, and the lost game must be named.
    """
    base = _report([_row("keep", 1, 100), _row("a", 0, 100), _row("b", 0, 100), _row("c", 0, 100)])
    arm = _report([_row("keep", 0, 100), _row("a", 1, 100), _row("b", 1, 100), _row("c", 1, 100)])

    cmp = ledger.compare(base, arm)
    ok, why = ledger.verdict(cmp)

    assert cmp["arm_total_levels"] > cmp["baseline_total_levels"], "the aggregate really is better"
    assert not ok, "and it is still refused"
    assert "REFUSED" in why and "keep" in why, "the game that broke must be named"


def test_promotion_needs_more_than_one_improved_game():
    base = _report([_row("a", 0, 100), _row("b", 0, 100)])
    arm = _report([_row("a", 1, 100), _row("b", 0, 100)])

    ok, why = ledger.verdict(ledger.compare(base, arm))

    assert not ok
    assert "HOLD" in why and "coincidence" in why


def test_two_improved_games_and_no_regression_promotes():
    base = _report([_row("a", 0, 100), _row("b", 0, 100), _row("c", 1, 100)])
    arm = _report([_row("a", 1, 100), _row("b", 1, 100), _row("c", 1, 100)])

    ok, why = ledger.verdict(ledger.compare(base, arm))

    assert ok and "PROMOTE" in why


def test_same_levels_but_strictly_cheaper_promotes_because_the_score_squares_efficiency():
    base = _report([_row("a", 1, 9000), _row("b", 1, 9000)])
    arm = _report([_row("a", 1, 400), _row("b", 1, 500)])

    ok, why = ledger.verdict(ledger.compare(base, arm))

    assert ok and "fewer actions" in why


def test_cheaper_on_one_game_and_costlier_on_another_does_not_promote():
    """Mixed efficiency is not an improvement, it is a trade nobody authorised."""
    base = _report([_row("a", 1, 9000), _row("b", 1, 100)])
    arm = _report([_row("a", 1, 400), _row("b", 1, 5000)])

    ok, _ = ledger.verdict(ledger.compare(base, arm))

    assert not ok


def test_a_game_that_errored_on_either_side_is_excluded_from_the_comparison():
    """An error is missing data. Reading it as a regression would refuse promotions for a flake.

    It is still visible: `aggregate` counts it in `games_errored`, so it cannot be lost entirely.
    """
    base = _report([_row("a", 1, 100), _row("b", 0, 100), _row("c", 0, 100)])
    arm = _report([_row("a", 0, 0, "timeout"), _row("b", 1, 100), _row("c", 1, 100)])

    cmp = ledger.compare(base, arm)
    ok, _ = ledger.verdict(cmp)

    assert cmp["levels_regressed"] == [], "the errored game is not a regression"
    assert ok, "the two genuine improvements still count"


# --------------------------------------------------------------------------- ledger safety


def test_promote_refuses_a_flag_that_was_never_measured(tmp_path, monkeypatch, capsys):
    """Promotion reads recorded evidence and never measures implicitly.

    A promote that quietly runs its own measurement is a promote nobody reviewed.
    """
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "l.yaml")

    rc = ledger.cmd_promote("CARNOT_ARC_NEVER_TESTED")

    assert rc == 1
    assert "no ledger entry" in capsys.readouterr().out


def test_promote_refuses_a_flag_whose_last_measurement_said_no(tmp_path, monkeypatch, capsys):
    import yaml

    lp = tmp_path / "l.yaml"
    lp.write_text(
        yaml.safe_dump(
            {
                "flags": {
                    "CARNOT_ARC_X": {
                        "state": "unevaluated",
                        "promotable": False,
                        "evidence": [{"verdict": "REFUSED: regressed on ['ka59']."}],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(ledger, "LEDGER", lp)

    rc = ledger.cmd_promote("CARNOT_ARC_X")

    assert rc == 1
    out = capsys.readouterr().out
    assert "REFUSING" in out and "ka59" in out, "must repeat the reason, not just decline"


def test_a_corrupt_ledger_raises_instead_of_starting_a_fresh_one(tmp_path, monkeypatch):
    """Silently replacing an unreadable ledger would erase every promotion and its evidence."""
    import pytest

    lp = tmp_path / "l.yaml"
    lp.write_text("{[: not yaml at all\n  - ][")
    monkeypatch.setattr(ledger, "LEDGER", lp)

    with pytest.raises(SystemExit):
        ledger.load()


def test_discover_reads_flags_from_source_not_a_hardcoded_list():
    """A hand-maintained watch-list drifts narrower than the thing it watches.

    That is this project's most-repeated defect, so the flag set is derived from the agent source.
    """
    flags = ledger.discover_flags()

    assert len(flags) > 50, f"expected the real agent flag surface, got {len(flags)}"
    assert all(f.startswith("CARNOT_ARC_") for f in flags)


def test_bench_report_always_carries_the_held_out_caveat():
    """The benchmark runs on PUBLIC games. Nobody may quote it as a hidden-game result."""
    assert "PUBLIC" in arc_bench.CAVEAT
    assert "not a hidden-game result" in arc_bench.CAVEAT


def test_roster_comes_from_the_registry_and_holds_the_public_games():
    games = arc_bench.roster()

    assert len(games) >= 25
    for known in ("ls20", "ka59", "dc22", "vc33"):
        assert known in games


def test_run_game_reports_an_error_row_rather_than_raising(monkeypatch):
    """A benchmark that dies on one game loses the whole sweep, including the games that passed."""
    import sys
    import types

    # Stubbed through sys.modules rather than imported: the real solver kit pulls the agent stack
    # in and trips the suite's memory-leak guard at ~500MB. The behaviour under test is this
    # function's error handling, not the arcade.
    kit = types.ModuleType("carnot.agentic.arc_solver_kit")

    def boom():
        raise RuntimeError("arcade unavailable")

    kit.offline_arcade = boom
    ge = types.ModuleType("carnot.agentic.arc_graph_explore")
    ge.graph_explore_solve_v2 = lambda *a, **k: (None, 0)
    monkeypatch.setitem(sys.modules, "carnot.agentic.arc_solver_kit", kit)
    monkeypatch.setitem(sys.modules, "carnot.agentic.arc_graph_explore", ge)

    row = arc_bench.run_game("ls20")

    assert row["error"] and "arcade unavailable" in row["error"]
    assert row["levels_cleared"] == 0
    assert row["game"] == "ls20"


def test_measured_report_round_trips_as_json():
    """The report is consumed by another process, so it must serialise."""
    rows = [_row("a", 1, 100), _row("b", 0, 200, "err")]
    report = {"schema": arc_bench.SCHEMA, "per_game_rows": rows, **arc_bench.aggregate(rows)}

    assert json.loads(json.dumps(report))["schema"] == "carnot.arc_bench.v1"


# --------------------------------------------------------------------------- reachability guard
#
# The benchmark drives `graph_explore_solve_v2`, not the full E3AgentPolicy cascade. Measured
# 2026-08-13: 48 of the 95 tracked flags are inside that import closure, 47 are not. Setting an
# unreachable flag produces a byte-identical sweep, which the promotion rule reads as HOLD -- "no
# level gained" -- filing a real capability as worthless with evidence attached. For 47 flags that
# verdict would be wrong, and wrong in the most damaging direction available.


def test_reachability_is_computed_transitively_not_from_the_entry_file_alone():
    """A direct-imports-only closure would report ~4 flags and refuse 91 real ones."""
    reach = ledger.reachable_flags()

    assert len(reach) > 20, f"closure looks shallow: {len(reach)} flags"
    # Read directly by the entry module...
    assert "CARNOT_ARC_SMALL_OBJECT_FIRST" in reach
    # ...and by a module several hops down the import chain.
    assert "CARNOT_ARC_INDUCE_MAX_TOKENS" in reach


def test_reachability_excludes_flags_outside_the_benchmark_path():
    """If everything were 'reachable' the guard would be decorative."""
    reach = ledger.reachable_flags()
    all_flags = set(ledger.discover_flags())

    assert reach < all_flags, "some flags must be unreachable or this guard does nothing"


def test_measure_refuses_an_unreachable_flag_and_explains_the_false_negative(
    tmp_path, monkeypatch, capsys
):
    """Refusing to measure is a smaller error than measuring the wrong thing confidently."""
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "l.yaml")
    monkeypatch.setattr(ledger, "reachable_flags", lambda *a, **k: {"CARNOT_ARC_REACHABLE"})

    called = {"n": 0}
    monkeypatch.setattr(ledger, "run_bench", lambda *a, **k: called.__setitem__("n", 1) or {})

    rc = ledger.cmd_measure("CARNOT_ARC_ELSEWHERE", "1")
    out = capsys.readouterr().out

    assert rc == 1
    assert called["n"] == 0, "it must not burn a sweep on a flag it cannot see"
    assert "REFUSING" in out
    assert "never" in out and "runs" in out, "must say WHY, not just decline"


def test_force_records_an_unreachable_null_deliberately(tmp_path, monkeypatch):
    """The escape hatch exists, but it has to be typed -- an accident cannot take it."""
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "l.yaml")
    monkeypatch.setattr(ledger, "reachable_flags", lambda *a, **k: set())
    rep = _report([_row("a", 1, 100)])
    monkeypatch.setattr(ledger, "run_bench", lambda *a, **k: rep)

    rc = ledger.cmd_measure("CARNOT_ARC_ELSEWHERE", "1", force=True)

    assert rc == 0
    import yaml

    saved = yaml.safe_load((tmp_path / "l.yaml").read_text())
    entry = saved["flags"]["CARNOT_ARC_ELSEWHERE"]
    assert entry["benchmark_reachable"] is False, (
        "a forced null must be stamped unreachable, so a later reader cannot mistake it for "
        "evidence that the capability does nothing"
    )


# --------------------------------------------------------------------------- the scored engine
#
# REQ-ARC-BENCH-6269. The explore engine reaches 48 of 95 flags; the other 47 are read by the
# cascade -- induction, the world-model verifier, planning, frontier tiers. The scored engine
# drives E3AgentPolicy, which is what `make_carnot_agent` builds and what the competition runs,
# and reaches 89. The remaining 6 are tooling knobs, not agent capabilities.


def test_scored_engine_reaches_far_more_flags_than_explore():
    """If the second engine did not widen coverage there would be no reason to have built it."""
    explore = ledger.reachable_flags(engine="explore")
    scored = ledger.reachable_flags(engine="scored")

    assert len(scored) > len(explore) + 30, f"explore={len(explore)} scored={len(scored)}"
    # A flag the explore engine provably cannot see, which is the whole motivating case.
    assert "CARNOT_ARC_HAZARD_MOVE_PRUNER" not in explore
    assert "CARNOT_ARC_HAZARD_MOVE_PRUNER" in scored


def test_every_engine_in_arc_bench_has_a_declared_entry_point():
    """A new engine whose entry point is not declared silently under-reports its own reach.

    Under-reporting reachability is the direction that files real capabilities as duds, so the two
    lists are pinned together rather than left to be kept in sync by memory.
    """
    import argparse
    import inspect

    src = inspect.getsource(arc_bench.main)
    # The engines arc_bench offers on the command line...
    offered = set()
    for line in src.splitlines():
        if "choices=[" in line and "explore" in line:
            offered = set(eval(line.split("choices=")[1].split("]")[0] + "]"))  # noqa: S307
    assert offered, "could not read arc_bench's engine choices"
    assert offered == set(ledger.ENGINE_ENTRY), (
        f"arc_bench offers {offered} but ENGINE_ENTRY declares {set(ledger.ENGINE_ENTRY)}; "
        "reachability for the undeclared engine would fall back to the explore closure"
    )
    del argparse


def test_refusal_names_the_engine_that_can_see_the_flag(tmp_path, monkeypatch, capsys):
    """A refusal that does not say what to do instead is a dead end.

    The whole point of the second engine is that a flag refused on `explore` is usually measurable
    on `scored`, so the message has to route the reader there.
    """
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "l.yaml")
    monkeypatch.setattr(ledger, "run_bench", lambda *a, **k: pytest_fail_if_called())

    def pytest_fail_if_called():
        raise AssertionError("must not run a sweep for a refused flag")

    rc = ledger.cmd_measure("CARNOT_ARC_HAZARD_MOVE_PRUNER", "1", engine="explore")
    out = capsys.readouterr().out

    assert rc == 1
    assert "--engine scored" in out


def _stub_scored_deps(monkeypatch, run_game_result=None, policy_raises=False):
    """Stand in for the two heavy modules `run_game_scored` imports.

    Importing `arc_competition_agent` and `arc_leaderboard_eval` for real pulls torch and the whole
    agent stack into the test process, which trips the suite's memory-leak guard at ~500MB. The
    behaviour under test is this function's own bookkeeping -- which action count it reports, and
    whether it restores the env var -- so the dependencies are stubbed through `sys.modules` and
    never loaded.
    """
    import sys
    import types

    def _policy(*a, **k):
        if policy_raises:
            raise RuntimeError("policy exploded")
        return object()

    aca = types.ModuleType("carnot.agentic.arc_competition_agent")
    aca.E3AgentPolicy = _policy
    lb = types.ModuleType("arc_leaderboard_eval")
    lb.run_game = lambda *a, **k: run_game_result or {}
    monkeypatch.setitem(sys.modules, "carnot.agentic.arc_competition_agent", aca)
    monkeypatch.setitem(sys.modules, "arc_leaderboard_eval", lb)


def test_scored_row_uses_charged_actions_not_raw_actions(monkeypatch):
    """The live gateway bills resets. Reporting the cheaper number flatters every comparison.

    Measured on vc33: actions=387, charged_actions=400. A benchmark reporting 387 would show an
    efficiency gain the competition would not pay out.
    """
    _stub_scored_deps(monkeypatch, {"levels": 2, "actions": 387, "charged_actions": 400})

    row = arc_bench.run_game_scored("vc33", budget=400)

    assert row["actions_spent"] == 400, "must bill resets the way the gateway does"
    assert row["levels_cleared"] == 2
    assert row["engine"] == "scored"


def test_scored_engine_restores_the_induction_env_var(monkeypatch):
    """This flips a process-global. Leaving it flipped changes every later cell in the sweep."""
    import os

    _stub_scored_deps(monkeypatch, {"levels": 0, "charged_actions": 1})
    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "sentinel")

    arc_bench.run_game_scored("vc33", budget=10, llm=True)

    assert os.environ["CARNOT_ARC_DISABLE_INDUCTION"] == "sentinel"


def test_scored_engine_reports_an_error_row_rather_than_raising(monkeypatch):
    """A sweep that dies on one game loses every game that already passed."""
    _stub_scored_deps(monkeypatch, policy_raises=True)

    row = arc_bench.run_game_scored("vc33", budget=10)

    assert row["error"] and "policy exploded" in row["error"]
    assert row["levels_cleared"] == 0


# --------------------------------------------------------------------------- fire counters
#
# REQ-ARC-BENCH-6270. Import reachability proves a flag's code is LOADED. It cannot prove the code
# RUNS. The first scored-engine A/B hit that gap: CARNOT_ARC_HAZARD_MOVE_PRUNER produced baseline
# and arm sweeps identical to the digit -- 16 levels, 48,313 actions, both arms. Read naively that
# is "measured, does not help." Measured directly on tu93 with the flag on: flag_resolved True,
# observe_calls 292, observed_nav_transitions 288, model_fitted FALSE. The lever is wired and
# running; the hypothesis class simply never fits. Three different facts, three different fixes.


def test_identical_arms_with_no_counter_movement_is_a_wiring_result_not_a_null():
    """The flag never resolved. Recording HOLD here files a capability as worthless."""
    rows = [_row("a", 1, 100), _row("b", 0, 200)]
    rep = _report(rows)

    ok, why = ledger.verdict(ledger.compare(rep, _report([dict(r) for r in rows])))

    assert not ok
    assert "UNINTERPRETABLE_NO_EFFECT" in why
    assert "WIRING" in why
    assert "Do NOT record it as a null" in why


def test_identical_arms_but_moved_counters_is_a_different_verdict():
    """Wired and running, outcome unchanged. Not the same claim as 'never fired'."""
    base = _report([dict(_row("a", 1, 100), fire_counters={"hz": {"observe_calls": 0}})])
    arm = _report([dict(_row("a", 1, 100), fire_counters={"hz": {"observe_calls": 288}})])

    cmp = ledger.compare(base, arm)
    ok, why = ledger.verdict(cmp)

    assert cmp["identical_to_baseline"] is True
    assert cmp["lever_counters_moved"] is True
    assert not ok
    assert "UNINTERPRETABLE_FIRED_NO_EFFECT" in why


def test_a_real_change_is_never_called_uninterpretable():
    """The guard must not swallow genuine results -- that would be the opposite failure."""
    base = _report([_row("a", 0, 100), _row("b", 0, 100)])
    arm = _report([_row("a", 1, 100), _row("b", 1, 100)])

    cmp = ledger.compare(base, arm)
    ok, why = ledger.verdict(cmp)

    assert cmp["identical_to_baseline"] is False
    assert ok and "PROMOTE" in why


def test_no_effect_is_checked_before_the_regression_and_hold_rules():
    """Ordering matters. 'Did nothing' outranks 'did not help enough' -- different instructions."""
    rows = [_row("a", 0, 100)]

    _, why = ledger.verdict(ledger.compare(_report(rows), _report([dict(r) for r in rows])))

    assert "UNINTERPRETABLE" in why
    assert "HOLD" not in why, "a no-op must never be reported as a measured null"


def test_fire_counters_are_discovered_by_scanning_not_from_a_name_list():
    """A hand-written lever list goes stale the next time a lever is added.

    A silently missing counter is exactly the false null this whole mechanism exists to prevent,
    so the collector reflects over the live object instead.
    """
    import inspect

    src = inspect.getsource(arc_bench._fire_counters)

    assert "dir(" in src, "must enumerate the object, not a maintained list"
    assert "_diagnostics" in src


def test_a_diagnostics_method_that_raises_does_not_lose_the_row():
    class Boom:
        def bad_diagnostics(self):
            raise RuntimeError("nope")

        def good_diagnostics(self):
            return {"observe_calls": 3}

    class Pol:
        explorer = Boom()

    out = arc_bench._fire_counters(Pol())

    assert out["explorer.good_diagnostics"] == {"observe_calls": 3}
    assert "diagnostics_error" in out["explorer.bad_diagnostics"]


# --------------------------------------------------------------------------- sweep safety
#
# REQ-ARC-FLAG-SWEEP-6271. Of the 54 default-off flags the scored engine reaches, only 20 are
# boolean capability toggles. A sweep that set every one to "1" would set GGUF_PATH to a nonsense
# path, INDUCE_TIMEOUT to one second, and DISABLE_INDUCTION to on -- turning the LLM OFF -- then
# record the resulting damage as evidence that those capabilities are harmful. A ledger full of
# confident wrong verdicts is worse than an empty one.


def test_a_path_flag_is_never_swept():
    assert ledger.classify_flag("CARNOT_ARC_GGUF_PATH") == "path"


def test_a_numeric_knob_is_never_swept():
    """`INDUCE_TIMEOUT=1` is a one-second timeout, which breaks induction rather than enabling it."""
    assert ledger.classify_flag("CARNOT_ARC_INDUCE_TIMEOUT") == "numeric"


def test_an_inverse_flag_is_never_swept():
    """Turning DISABLE_INDUCTION on removes the LLM. A regression would be the expected result."""
    assert ledger.classify_flag("CARNOT_ARC_DISABLE_INDUCTION") == "inverse"


def test_a_write_permission_flag_is_never_swept():
    """E3_ALLOW_EVIDENCE_WRITE disables the guard stopping tests writing the tracked record.

    Inert outside pytest, so sweeping it yields a confident meaningless null -- and its presence
    in a list called "capability flags" would tell the next reader it had been measured and found
    worthless.
    """
    assert ledger.classify_flag("CARNOT_ARC_E3_ALLOW_EVIDENCE_WRITE") == "guard"


def test_a_genuine_boolean_capability_is_swept():
    """The classifier must not be so cautious that nothing is measurable."""
    assert ledger.classify_flag("CARNOT_ARC_ACTIVE_PROBE") == "bool"


def test_sweep_reports_every_exclusion_rather_than_narrowing_silently(tmp_path, monkeypatch):
    """A sweep that quietly shrinks its candidate list is the project's signature bug with a
    progress bar."""
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "l.yaml")

    todo, skipped = ledger.sweep_candidates("scored", {"flags": {}})

    assert todo, "some flags must be measurable"
    assert len(skipped) > len(todo), "most flags are excluded, and each exclusion carries a reason"
    assert all(v for v in skipped.values()), "every skip states why"
    assert any("corrupt" in v for v in skipped.values()), "value-knob exclusions explain the risk"


def test_sweep_is_resumable_and_skips_what_it_already_measured(tmp_path, monkeypatch):
    """An interrupted six-hour sweep must continue, not restart."""
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "l.yaml")
    todo_before, _ = ledger.sweep_candidates("scored", {"flags": {}})
    done = todo_before[0]

    todo_after, skipped = ledger.sweep_candidates(
        "scored", {"flags": {done: {"evidence": [{"verdict": "HOLD"}]}}}
    )

    assert done not in todo_after
    assert skipped[done] == "already measured"


def test_sweep_dry_run_measures_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "l.yaml")
    monkeypatch.setattr(
        ledger, "run_bench", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran a sweep"))
    )

    assert ledger.cmd_sweep("scored", None, dry_run=True) == 0
