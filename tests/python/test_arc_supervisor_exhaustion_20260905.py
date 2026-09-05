"""Spec: REQ-ARC-WMTE-7030, 7031, 7032, 7033 and their scenarios.

The trajectory supervisor says which arms its run could fire and what it saw at each window
it could not answer; the refinement tool reads both, keeps shadow receipts as controls, and
decides exhaustion PER LEVEL (the axis the arms reset on), never pooled over the run.

MEASURED 2026-09-05 (population: the 14 receipts in ops/arc_supervisor_refinement_ledger.json
and the 14 artifacts in results/arc_leaderboard_eval_runs/):

- All 64 unredirected windows sit in 5 eval receipts where the three default-on arms all
  fired. Only ONE of those runs had the env-gated tool rung on, so a trigger that waits for
  `set(ARM_ORDER)` reported one cell and hid four (53 of 64 windows). REQ-7030.
- No receipt says WHY the table ran dry (attempt cap? diversity already on?). REQ-7031.
- One shadow receipt (r11l, seed 20260719, window 120) leveled up at action 813 with nothing
  applied, and its would-have rows at 120/240/360 carry the same 765/645/525 actions-to-level-up
  that four applied runs booked as `helped`. A control now exists; the tool ignored it. REQ-7032.
- The five cells the pooled trigger emitted were all false: `_arms_used` is cleared on every
  level-up (arc_trajectory_supervisor.observe), so "every arm fired" pooled over the run is not
  "every arm spent on the level that stagnated". Read per level from the receipts alone, none of
  the 14 ledger rows is decidable (no window rows). REQ-7033.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.agentic.arc_supervisor_refinement import (
    LEGACY_DEFAULT_ARMS,
    MIN_FIRED_PER_ARM,
    NOT_DECIDABLE_NO_WINDOW_ROWS,
    NOT_DECIDABLE_ROWS_DROPPED,
    STATUS_INSUFFICIENT,
    STATUS_RECOMMENDATION,
    empty_ledger,
    enabled_arms_for_entry,
    evaluate,
    exhausted_windows_by_level,
    exhaustion_summary,
    ingest_files,
    load_ledger,
    render_report,
)
from carnot.agentic.arc_trajectory_supervisor import (
    ARM_ALLOW_REINDUCTION,
    ARM_DROP_GOAL_BIAS,
    ARM_FORCE_DIVERSITY,
    ARM_ORDER,
    ARM_TOOL_LOOP_REINDUCTION,
    MAX_UNREDIRECTED_WINDOWS,
    TrajectorySnapshot,
    TrajectorySupervisor,
    enabled_arms,
)

NOW = "2026-09-05T05:00:00+00:00"


def _snap(**overrides) -> TrajectorySnapshot:
    base = {
        "level": 0,
        "goal_bias_installed": False,
        "induced": False,
        "induction_attempts": 0,
        "new_transitions_since_induction": 0,
        "diversity_active": False,
    }
    base.update(overrides)
    return TrajectorySnapshot(**base)


def _redirect(
    arm: str, resolved: bool, level: int = 0, a2l: int | None = None, idx: int = 120
) -> dict:
    return {
        "arm": arm,
        "action_index": idx,
        "level": level,
        "diagnosis": "test",
        "resolved_by_levelup": resolved,
        "actions_to_levelup": a2l,
    }


def _applied_row(
    *,
    game: str = "r11l",
    seed: int = 20260719,
    redirects: list[dict] | None = None,
    stag: int = 0,
    arms_enabled: list[str] | None = None,
    windows: list[dict] | None = None,
    dropped: int = 0,
    levels: int = 2,
) -> dict:
    receipt: dict = {
        "enabled": True,
        "mode": "applied",
        "window": 120,
        "actions_observed": 2310,
        "arms_used": [],
        "arm_outcomes": {},
        "stagnations_unredirected": stag,
        "redirects": redirects or [],
    }
    if arms_enabled is not None:
        receipt["arms_enabled"] = arms_enabled
    if windows is not None:
        receipt["unredirected_windows"] = windows
        receipt["unredirected_windows_dropped"] = dropped
    return {
        "game": game,
        "seed": seed,
        "arm": "eval:e3:budget20000",
        "levels": levels,
        "actions": 2121,
        "trajectory_supervisor": receipt,
    }


def _shadow_row(
    *,
    game: str = "r11l",
    seed: int = 20260719,
    would_have: list[dict] | None = None,
) -> dict:
    return {
        "game": game,
        "seed": seed,
        "arm": "eval:e3:budget20000",
        "levels": 2,
        "actions": 2121,
        "trajectory_supervisor": {
            "enabled": False,
            "mode": "shadow",
            "window": 120,
            "actions_observed": 2310,
            "arms_used": [],
            "would_have_arm_outcomes": {},
            "stagnations_unredirected": 12,
            "would_have_redirects": would_have or [],
        },
    }


def _would_have(arm: str, followed: bool, level: int = 0, a2l: int | None = None) -> dict:
    return {
        "arm": arm,
        "action_index": 120,
        "level": level,
        "diagnosis": "test",
        "levelup_followed_without_redirect": followed,
        "actions_to_levelup_without_redirect": a2l,
        "co_credited_count": None,
    }


def _ingest(tmp_path: Path, rows: list[dict], name: str = "rows.json") -> tuple[dict, dict]:
    path = tmp_path / name
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    ledger = empty_ledger()
    counts = ingest_files(ledger, [path], NOW)
    return ledger, counts


def _windows(n: int, *, start: int = 1490, **state) -> list[dict]:
    """`n` window rows, 120 actions apart, on level 2 with two arms spent unless overridden."""
    base = {
        "level": 2,
        "arms_used": [ARM_DROP_GOAL_BIAS, ARM_ALLOW_REINDUCTION],
        "goal_bias_installed": False,
        "induced": True,
        "induction_attempts": 3,
        "attempt_cap_reached": True,
        "new_transitions_since_induction": 250,
        "evidence_floor_met": True,
        "diversity_active": True,
    }
    base.update(state)
    return [{"action_index": start + 120 * i, **base} for i in range(n)]


def _entry(ledger: dict) -> dict:
    return next(iter(ledger["entries"].values()))


# --- REQ-ARC-WMTE-7030: the receipt names the arms the run could fire ----------------------


def test_scenario_7030_a_enabled_arms_follow_the_tool_gate(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-7030-A: with the tool rung off, three arms are enabled; with it
    on, four. Configuration, not eligibility-by-state."""
    monkeypatch.delenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", raising=False)
    assert enabled_arms() == (ARM_DROP_GOAL_BIAS, ARM_ALLOW_REINDUCTION, ARM_FORCE_DIVERSITY)
    monkeypatch.setenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", "1")
    assert enabled_arms() == ARM_ORDER


def test_scenario_7030_b_receipt_carries_arms_enabled_in_order(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-7030-B: the receipt lists `arms_enabled` in ARM_ORDER order."""
    monkeypatch.delenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", raising=False)
    sup = TrajectorySupervisor(window=2)
    receipt = sup.receipt()
    assert receipt["arms_enabled"] == [
        ARM_DROP_GOAL_BIAS,
        ARM_ALLOW_REINDUCTION,
        ARM_FORCE_DIVERSITY,
    ]
    assert ARM_TOOL_LOOP_REINDUCTION not in receipt["arms_enabled"]


def test_scenario_7030_b_an_arm_that_fired_is_enabled_even_after_a_late_env_flip(
    monkeypatch,
) -> None:
    """SCENARIO-ARC-WMTE-7030-B: the set is a snapshot at construction unioned with what
    fired, so a mid-run env change cannot hide a firing from the exhaustion reader."""
    monkeypatch.delenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", raising=False)
    sup = TrajectorySupervisor(window=1, reinduction_evidence_floor=0)
    assert sup.observe(_snap(goal_bias_installed=True)).arm == ARM_DROP_GOAL_BIAS
    assert sup.observe(_snap(induced=True, induction_attempts=1)).arm == ARM_ALLOW_REINDUCTION
    monkeypatch.setenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", "1")
    fired = sup.observe(_snap(induced=True, induction_attempts=1))
    assert fired is not None and fired.arm == ARM_TOOL_LOOP_REINDUCTION
    assert sup.receipt()["arms_enabled"] == list(ARM_ORDER)


def test_scenario_7030_c_a_declared_four_arm_run_that_spent_three_on_a_level_is_not_exhausted(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7030-C as corrected by REQ-7033: the enabled set is still the
    receipt's `arms_enabled` (legacy rows fall back to the three default-on arms), but it is
    compared against the arms spent ON THE LEVEL of each window row. A run that declares the
    tool rung enabled and spent three arms on the level still had a rung left: not a cell. A
    run that declares three and spent three on the level IS a cell, source `receipt`."""
    spent_three = _windows(2, level=0, arms_used=list(LEGACY_DEFAULT_ARMS))
    declared_four = _applied_row(
        seed=1,
        redirects=[_redirect(arm, False) for arm in LEGACY_DEFAULT_ARMS],
        stag=2,
        arms_enabled=list(ARM_ORDER),
        windows=spent_three,
    )
    declared_three = _applied_row(
        seed=2,
        redirects=[_redirect(arm, False) for arm in LEGACY_DEFAULT_ARMS],
        stag=2,
        arms_enabled=list(LEGACY_DEFAULT_ARMS),
        windows=spent_three,
    )
    ledger, _ = _ingest(tmp_path, [declared_four, declared_three])
    spec = evaluate(ledger, NOW)["new_arm_specification"]
    assert spec is not None
    assert [(c["seed"], c["arms_enabled_source"], c["level"]) for c in spec["cells"]] == [
        (2, "receipt", 0)
    ]
    assert "REQ-ARC-WMTE-7033" in spec["trigger"]


def test_scenario_7030_c_a_fired_arm_is_unioned_into_a_declared_set(tmp_path: Path) -> None:
    """A receipt that declares three arms but fired the tool rung anyway (a late env flip
    on an old receipt) counts the tool rung as enabled, so a level is exhausted only when
    every one of the four was spent on it."""
    redirects = [_redirect(arm, False) for arm in ARM_ORDER]
    three_spent = _applied_row(
        seed=1,
        redirects=redirects,
        stag=1,
        arms_enabled=list(LEGACY_DEFAULT_ARMS),
        windows=_windows(1, level=0, arms_used=list(LEGACY_DEFAULT_ARMS)),
    )
    four_spent = _applied_row(
        seed=2,
        redirects=redirects,
        stag=1,
        arms_enabled=list(LEGACY_DEFAULT_ARMS),
        windows=_windows(1, level=0, arms_used=list(ARM_ORDER)),
    )
    ledger, _ = _ingest(tmp_path, [three_spent, four_spent])
    for entry in ledger["entries"].values():
        enabled, _ = enabled_arms_for_entry(entry)
        assert enabled == set(ARM_ORDER)
    spec = evaluate(ledger, NOW)["new_arm_specification"]
    assert spec is not None
    assert [c["seed"] for c in spec["cells"]] == [2]


def test_scenario_7030_c_a_legacy_row_falls_back_to_the_three_default_arms(
    tmp_path: Path,
) -> None:
    """A row with no `arms_enabled` whose window row spent the three default-on arms IS a
    cell; reading ARM_ORDER (which holds the default-off tool rung) as the set would hide it.
    The 2026-09-05 count: 4 of 5 pooled cells hidden by exactly that read."""
    row = _applied_row(
        redirects=[_redirect(arm, True, a2l=700) for arm in LEGACY_DEFAULT_ARMS],
        stag=1,
        windows=_windows(1, level=0, arms_used=list(LEGACY_DEFAULT_ARMS)),
    )
    ledger, _ = _ingest(tmp_path, [row])
    entry = _entry(ledger)
    assert entry["arms_enabled"] is None
    enabled, source = enabled_arms_for_entry(entry)
    assert enabled == set(LEGACY_DEFAULT_ARMS)
    assert source == "legacy_default"
    spec = evaluate(ledger, NOW)["new_arm_specification"]
    assert spec is not None
    assert [(c["level"], c["arms_enabled_source"]) for c in spec["cells"]] == [
        (0, "legacy_default")
    ]


# --- REQ-ARC-WMTE-7031: every exhausted window records the state the table saw ------------


def test_scenario_7031_a_an_exhausted_window_records_its_state(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-7031-A: the row names the spent arms and why the rest were
    ineligible (attempt cap reached, evidence floor, diversity already on)."""
    monkeypatch.delenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", raising=False)
    sup = TrajectorySupervisor(window=1, reinduction_evidence_floor=200, reinduction_attempt_cap=3)
    assert sup.observe(_snap(level=2, goal_bias_installed=True)).arm == ARM_DROP_GOAL_BIAS
    # Nothing left that is eligible: bias gone, attempts at the cap, diversity already on.
    # Same level as the firing above: a level change is progress, not stagnation.
    exhausted = _snap(
        level=2,
        induced=True,
        induction_attempts=3,
        new_transitions_since_induction=250,
        diversity_active=True,
    )
    assert sup.observe(exhausted) is None
    receipt = sup.receipt()
    assert receipt["stagnations_unredirected"] == 1
    assert receipt["unredirected_windows_dropped"] == 0
    assert receipt["unredirected_windows"] == [
        {
            "action_index": 2,
            "level": 2,
            "arms_used": [ARM_DROP_GOAL_BIAS],
            "goal_bias_installed": False,
            "induced": True,
            "induction_attempts": 3,
            "attempt_cap_reached": True,
            "new_transitions_since_induction": 250,
            "evidence_floor_met": True,
            "diversity_active": True,
        }
    ]


def test_scenario_7031_a_a_window_that_fires_records_no_row() -> None:
    """A window answered by an arm is not an exhausted window."""
    sup = TrajectorySupervisor(window=1)
    assert sup.observe(_snap(goal_bias_installed=True)) is not None
    assert sup.receipt()["unredirected_windows"] == []


def test_scenario_7031_a_the_cap_and_floor_flags_read_the_supervisor_settings() -> None:
    """`attempt_cap_reached` and `evidence_floor_met` use THIS supervisor's settings, so a
    reader does not have to know the cap to read the row."""
    sup = TrajectorySupervisor(window=1, reinduction_evidence_floor=10, reinduction_attempt_cap=5)
    assert (
        sup.observe(
            _snap(
                induced=True,
                induction_attempts=2,
                new_transitions_since_induction=4,
                diversity_active=True,
            )
        )
        is None
    )
    row = sup.receipt()["unredirected_windows"][0]
    assert row["attempt_cap_reached"] is False
    assert row["evidence_floor_met"] is False


def test_scenario_7031_b_the_window_list_is_bounded_and_overflow_is_counted() -> None:
    """SCENARIO-ARC-WMTE-7031-B: past MAX_UNREDIRECTED_WINDOWS rows, the count keeps
    growing and `unredirected_windows_dropped` says how many rows were not kept."""
    sup = TrajectorySupervisor(window=1)
    busy = _snap(diversity_active=True)
    for _ in range(MAX_UNREDIRECTED_WINDOWS + 5):
        assert sup.observe(busy) is None
    receipt = sup.receipt()
    assert receipt["stagnations_unredirected"] == MAX_UNREDIRECTED_WINDOWS + 5
    assert len(receipt["unredirected_windows"]) == MAX_UNREDIRECTED_WINDOWS
    assert receipt["unredirected_windows_dropped"] == 5


def test_scenario_7031_c_the_cell_summarises_the_states_on_its_own_level(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7031-C as corrected by REQ-7033: the ledger keeps every row; the
    cell counts only the rows on ITS level that spent every enabled arm. Nine level-2 rows
    with two arms spent are recorded but are not exhaustion; two level-0 rows with all three
    spent are the cell. The whole-entry summary still reads all eleven."""
    windows = _windows(9) + _windows(
        2,
        level=0,
        start=480,
        arms_used=list(LEGACY_DEFAULT_ARMS),
        attempt_cap_reached=False,
        induction_attempts=1,
    )
    row = _applied_row(
        redirects=[_redirect(arm, True, a2l=700) for arm in LEGACY_DEFAULT_ARMS],
        stag=11,
        arms_enabled=list(LEGACY_DEFAULT_ARMS),
        windows=windows,
    )
    ledger, _ = _ingest(tmp_path, [row])
    entry = _entry(ledger)
    assert len(entry["unredirected_windows"]) == 11
    assert entry["unredirected_windows_dropped"] == 0
    assert exhaustion_summary(entry)["windows_recorded"] == 11
    recommendation = evaluate(ledger, NOW)
    cells = recommendation["new_arm_specification"]["cells"]
    assert [c["level"] for c in cells] == [0]
    assert cells[0]["exhausted_windows"] == 2
    assert cells[0]["windows_on_level"] == 2
    assert cells[0]["exhaustion_states"] == {
        "windows_recorded": 2,
        "windows_dropped": 0,
        "levels": [0],
        "arms_used_sets": [",".join(LEGACY_DEFAULT_ARMS)],
        "goal_bias_installed": 0,
        "induced": 2,
        "attempt_cap_reached": 0,
        "evidence_floor_met": 2,
        "diversity_active": 2,
    }
    report = render_report(recommendation)
    assert "level=0 exhausted_windows=2" in report
    assert "states: windows=2 levels=[0] attempt_cap_reached=0" in report


def test_scenario_7031_c_a_legacy_row_reads_not_recorded_and_is_not_a_cell(
    tmp_path: Path,
) -> None:
    """A row from before the field must not read as "every flag False", and (REQ-7033) it
    must not read as a cell either: its levels cannot be told apart."""
    row = _applied_row(redirects=[_redirect(arm, False) for arm in LEGACY_DEFAULT_ARMS], stag=14)
    ledger, _ = _ingest(tmp_path, [row])
    entry = _entry(ledger)
    assert entry["unredirected_windows"] == []
    assert exhaustion_summary(entry) == "not_recorded"
    assert exhausted_windows_by_level(entry) == {}
    recommendation = evaluate(ledger, NOW)
    assert recommendation["new_arm_specification"] is None
    report = render_report(recommendation)
    assert "NEW ARM SPECIFICATION" not in report
    assert "reason=no_window_rows_recorded" in report


def test_scenario_7031_c_the_ledger_copy_is_bounded_too(tmp_path: Path) -> None:
    """An oversized producer list cannot bloat the ledger past the receipt's own cap."""
    row = _applied_row(
        redirects=[_redirect(arm, False) for arm in LEGACY_DEFAULT_ARMS],
        stag=MAX_UNREDIRECTED_WINDOWS + 10,
        windows=_windows(MAX_UNREDIRECTED_WINDOWS + 10),
    )
    ledger, _ = _ingest(tmp_path, [row])
    entry = _entry(ledger)
    assert len(entry["unredirected_windows"]) == MAX_UNREDIRECTED_WINDOWS


# --- REQ-ARC-WMTE-7032: shadow receipts are controls, never redirect evidence -------------


def test_scenario_7032_a_shadow_receipts_land_in_controls_not_entries(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7032-A: REQ-6720 rule 1 still holds (no shadow row in `entries`),
    and the shadow row is now kept in `controls` with its would-have outcomes."""
    shadow = _shadow_row(would_have=[_would_have(ARM_DROP_GOAL_BIAS, True, a2l=765)])
    ledger, counts = _ingest(tmp_path, [shadow, shadow])
    assert ledger["entries"] == {}
    assert counts["shadow_observed"] == 2
    assert counts["controls_new"] == 1
    assert counts["controls_duplicate"] == 1
    control = next(iter(ledger["controls"].values()))
    assert control["mode"] == "shadow"
    assert control["would_have_redirects"] == [
        {
            "arm": ARM_DROP_GOAL_BIAS,
            "action_index": 120,
            "level": 0,
            "levelup_followed_without_redirect": True,
            "actions_to_levelup_without_redirect": 765,
        }
    ]
    # Redirect evidence is unchanged by the control pool.
    assert evaluate(ledger, NOW)["evidence"]["redirects"] == 0
    assert evaluate(ledger, NOW)["evidence"]["controls"] == 1


def test_scenario_7032_b_a_credit_the_control_reproduces_is_matched(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7032-B: the 2026-09-05 r11l shape. Four applied runs booked
    `drop_goal_bias` as helped at level 0; a shadow run in the same cell leveled up with
    nothing applied. Those credits are control-matched; a credit in a cell with no control,
    or where the control did NOT level up, is not."""
    applied = [
        _applied_row(seed=20260719, redirects=[_redirect(ARM_DROP_GOAL_BIAS, True, a2l=765)]),
        _applied_row(
            seed=20260719,
            redirects=[
                _redirect(ARM_DROP_GOAL_BIAS, True, a2l=768),
                _redirect(ARM_FORCE_DIVERSITY, True, level=1, a2l=5),
            ],
        ),
        _applied_row(
            game="cd82", seed=20260719, redirects=[_redirect(ARM_DROP_GOAL_BIAS, True, a2l=651)]
        ),
        _applied_row(seed=99, redirects=[_redirect(ARM_ALLOW_REINDUCTION, True, a2l=100)]),
    ]
    shadow = [
        _shadow_row(
            seed=20260719,
            would_have=[
                _would_have(ARM_DROP_GOAL_BIAS, True, a2l=765),
                _would_have(ARM_FORCE_DIVERSITY, False, level=1),
            ],
        ),
        _shadow_row(seed=99, would_have=[_would_have(ARM_ALLOW_REINDUCTION, False)]),
    ]
    ledger, _ = _ingest(tmp_path, applied + shadow)
    recommendation = evaluate(ledger, NOW)
    per_arm = {row["arm"]: row for row in recommendation["per_arm"]}
    assert per_arm[ARM_DROP_GOAL_BIAS]["helped"] == 3
    assert per_arm[ARM_DROP_GOAL_BIAS]["helped_matched_by_control"] == 2
    assert per_arm[ARM_DROP_GOAL_BIAS]["helped_beyond_control"] == 1
    # The level-1 diversity credit: control exists but did not level up there.
    assert per_arm[ARM_FORCE_DIVERSITY]["helped_matched_by_control"] == 0
    # A different seed is a different cell.
    assert per_arm[ARM_ALLOW_REINDUCTION]["helped_matched_by_control"] == 0
    assert recommendation["evidence"]["helped_total"] == 5
    assert recommendation["evidence"]["helped_matched_by_control_total"] == 2
    assert recommendation["evidence"]["controls"] == 2
    report = render_report(recommendation)
    assert "controls: 2 shadow receipts; 2 of 5 credits matched by a control" in report
    assert f"{ARM_DROP_GOAL_BIAS}: fired=3 helped=3" in report
    assert "beyond_control=1" in report


def test_scenario_7032_b_an_unhelped_redirect_is_never_matched(tmp_path: Path) -> None:
    """A redirect that no level-up credited cannot be "reproduced" by a control."""
    ledger, _ = _ingest(
        tmp_path,
        [
            _applied_row(redirects=[_redirect(ARM_DROP_GOAL_BIAS, False)]),
            _shadow_row(would_have=[_would_have(ARM_DROP_GOAL_BIAS, True, a2l=765)]),
        ],
    )
    row = evaluate(ledger, NOW)["per_arm"][0]
    assert row["arm"] == ARM_DROP_GOAL_BIAS
    assert (row["helped"], row["helped_matched_by_control"], row["helped_beyond_control"]) == (
        0,
        0,
        0,
    )


def test_scenario_7032_c_the_frozen_rules_still_key_on_pooled_helped(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7032-C: an arm whose every credit a control reproduces is still not
    a retire candidate. Changing a rule is a separate spec act (as REQ-7013-D recorded)."""
    applied = [
        _applied_row(seed=i, redirects=[_redirect(ARM_DROP_GOAL_BIAS, True, a2l=700)])
        for i in range(MIN_FIRED_PER_ARM)
    ]
    shadow = [
        _shadow_row(seed=i, would_have=[_would_have(ARM_DROP_GOAL_BIAS, True, a2l=700)])
        for i in range(MIN_FIRED_PER_ARM)
    ]
    ledger, _ = _ingest(tmp_path, applied + shadow)
    recommendation = evaluate(ledger, NOW)
    row = recommendation["per_arm"][0]
    assert row["helped"] == MIN_FIRED_PER_ARM
    assert row["helped_beyond_control"] == 0
    assert [r for r in recommendation["recommendations"] if r["kind"] == "retire_candidate"] == []


def test_scenario_7032_d_a_ledger_from_before_the_control_pool_loads(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7032-D: the v1 ledger on disk has no `controls`; it loads with an
    empty pool rather than failing or reading a missing key."""
    ledger = empty_ledger()
    del ledger["controls"]
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger), encoding="utf-8")
    loaded = load_ledger(path)
    assert loaded["controls"] == {}
    recommendation = evaluate(loaded, NOW)
    assert recommendation["evidence"]["controls"] == 0


# --- REQ-ARC-WMTE-7033: exhaustion is decided on the axis the arms reset on ---------------


def test_scenario_7033_a_arms_spent_on_different_levels_are_not_exhaustion(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7033-A: the 2026-09-05 defect input. One arm fired on each of
    levels 0, 1 and 2, each credited by that level's level-up, then two windows passed on
    level 3 with NO arm spent. Pooled over the run every enabled arm fired and stagnation
    continued; on no single level was the table ever out of arms. Not a cell, and (rows
    exist) not undecidable either."""
    row = _applied_row(
        redirects=[
            _redirect(ARM_DROP_GOAL_BIAS, True, level=0, a2l=200, idx=120),
            _redirect(ARM_ALLOW_REINDUCTION, True, level=1, a2l=150, idx=440),
            _redirect(ARM_FORCE_DIVERSITY, True, level=2, a2l=90, idx=710),
        ],
        stag=2,
        arms_enabled=list(LEGACY_DEFAULT_ARMS),
        windows=_windows(2, level=3, start=920, arms_used=[]),
        levels=3,
    )
    ledger, _ = _ingest(tmp_path, [row])
    recommendation = evaluate(ledger, NOW)
    assert recommendation["new_arm_specification"] is None
    assert recommendation["exhaustion_not_decidable"] == []
    assert recommendation["evidence"]["exhaustion_not_decidable"] == 0


def _replay_r11l_1408494(monkeypatch) -> TrajectorySupervisor:
    """Drive the REAL supervisor through the redirect and level-up timeline recorded in
    results/arc_leaderboard_eval_runs/cd82-r11l-1408494.partial.json (r11l, seed 20260719,
    window 120, 2310 observations): drop_goal_bias at 120, force_exploration_diversity at 240,
    allow_reinduction at 360, level-ups at 885 and 1010, drop_goal_bias at 1130 and
    allow_reinduction at 1250 on level 2, 13 unredirected windows. The snapshot stream is
    synthetic; the firing sequence, level-up calls and window count it produces are the
    recorded ones, checked by the assertions in the caller."""
    monkeypatch.delenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", raising=False)
    sup = TrajectorySupervisor(
        window=120, reinduction_evidence_floor=200, reinduction_attempt_cap=3
    )
    level_ups = {885: 1, 1010: 2}

    def snapshot(t: int) -> TrajectorySnapshot:
        if t <= 120:
            return _snap(level=0, goal_bias_installed=True)
        if t <= 240:
            return _snap(level=0)
        if t <= 360:
            return _snap(
                level=0, induced=True, induction_attempts=1, new_transitions_since_induction=299
            )
        if t < 885:
            return _snap(level=0, induced=True, induction_attempts=2, diversity_active=True)
        if t < 1010:
            return _snap(level=1, diversity_active=True)
        if t <= 1130:
            return _snap(level=2, goal_bias_installed=True, diversity_active=True)
        if t <= 1250:
            return _snap(
                level=2,
                induced=True,
                induction_attempts=1,
                new_transitions_since_induction=223,
                diversity_active=True,
            )
        return _snap(level=2, induced=True, induction_attempts=3, diversity_active=True)

    for t in range(1, 2311):
        level = level_ups.get(t)
        snap = snapshot(t) if level is None else _snap(level=level, diversity_active=True)
        sup.observe(snap)
    return sup


def test_scenario_7033_b_the_recorded_r11l_timeline_is_exhausted_on_level_0_only(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-7033-B: the input that fires the corrected cell is reachable, and
    it is the recorded r11l run. Replayed through the real supervisor the receipt shows the
    recorded 13 windows: four on level 0 after all three arms were spent (480/600/720/840),
    one on level 1 with nothing spent, eight on level 2 with two arms spent. Exactly one cell:
    level 0, resolved by the level-up at 885, 405 actions after the first exhausted window.
    Level 2 -- the level that never resolved -- is NOT a cell: one rung was never spent there."""
    sup = _replay_r11l_1408494(monkeypatch)
    receipt = sup.receipt()
    assert [(r["arm"], r["action_index"], r["level"]) for r in receipt["redirects"]] == [
        (ARM_DROP_GOAL_BIAS, 120, 0),
        (ARM_FORCE_DIVERSITY, 240, 0),
        (ARM_ALLOW_REINDUCTION, 360, 0),
        (ARM_DROP_GOAL_BIAS, 1130, 2),
        (ARM_ALLOW_REINDUCTION, 1250, 2),
    ]
    assert [r["actions_to_levelup"] for r in receipt["redirects"]] == [765, 645, 525, None, None]
    assert receipt["stagnations_unredirected"] == 13
    rows = receipt["unredirected_windows"]
    assert [(w["action_index"], w["level"]) for w in rows] == [
        (480, 0),
        (600, 0),
        (720, 0),
        (840, 0),
        (1005, 1),
        *[(1370 + 120 * i, 2) for i in range(8)],
    ]
    assert all(set(w["arms_used"]) == set(LEGACY_DEFAULT_ARMS) for w in rows[:4])
    assert rows[4]["arms_used"] == []
    assert all(
        w["arms_used"] == sorted([ARM_ALLOW_REINDUCTION, ARM_DROP_GOAL_BIAS]) for w in rows[5:]
    )

    row = {
        "game": "r11l",
        "seed": 20260719,
        "arm": "eval:e3:budget20000",
        "levels": 2,
        "actions": 2126,
        "trajectory_supervisor": {**receipt, "mode": "applied"},
    }
    ledger, _ = _ingest(tmp_path, [row])
    entry = _entry(ledger)
    assert set(exhausted_windows_by_level(entry)) == {0}
    recommendation = evaluate(ledger, NOW)
    cells = recommendation["new_arm_specification"]["cells"]
    assert len(cells) == 1
    cell = cells[0]
    assert cell["level"] == 0
    assert cell["exhausted_windows"] == 4
    assert cell["windows_on_level"] == 4
    assert cell["first_exhausted_action_index"] == 480
    assert cell["arms_fired_on_level"] == sorted(LEGACY_DEFAULT_ARMS)
    assert cell["arms_enabled"] == sorted(LEGACY_DEFAULT_ARMS)
    assert cell["arms_enabled_source"] == "receipt"
    assert cell["level_resolved_by_levelup"] is True
    assert cell["actions_from_first_exhaustion_to_levelup"] == 405
    assert cell["exhaustion_states"]["windows_recorded"] == 4
    assert cell["exhaustion_states"]["levels"] == [0]
    assert recommendation["exhaustion_not_decidable"] == []
    report = render_report(recommendation)
    assert "level=0 exhausted_windows=4 level_resolved_by_levelup=True" in report
    assert "actions_from_first_exhaustion_to_levelup=405" in report


def test_scenario_7033_b_a_level_that_never_resolves_reads_unresolved(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7033-B: a cell on a level no level-up ever cleared reads
    `level_resolved_by_levelup: false` and no actions-to-level-up, and names its level."""
    row = _applied_row(
        redirects=[
            _redirect(arm, False, level=2, idx=1130 + 120 * i)
            for i, arm in enumerate(LEGACY_DEFAULT_ARMS)
        ],
        stag=3,
        arms_enabled=list(LEGACY_DEFAULT_ARMS),
        windows=_windows(3, level=2, start=1490, arms_used=list(LEGACY_DEFAULT_ARMS)),
    )
    ledger, _ = _ingest(tmp_path, [row])
    cells = evaluate(ledger, NOW)["new_arm_specification"]["cells"]
    assert [(c["level"], c["level_resolved_by_levelup"]) for c in cells] == [(2, False)]
    assert cells[0]["actions_from_first_exhaustion_to_levelup"] is None
    assert cells[0]["first_exhausted_action_index"] == 1490
    assert cells[0]["exhausted_windows"] == 3


def test_scenario_7033_c_a_stagnating_legacy_row_is_listed_as_not_decidable(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7033-C: the shape of every stagnating row in the 2026-09-05 ledger.
    Three default arms fired, 14 unredirected windows, no window rows. Not a cell (the old
    SCENARIO-7030-C said it was; retracted). Listed, with the reason, so zero cells cannot be
    read as "no level ever ran dry"."""
    row = _applied_row(
        redirects=[_redirect(arm, True, a2l=700) for arm in LEGACY_DEFAULT_ARMS],
        stag=14,
    )
    ledger, _ = _ingest(tmp_path, [row])
    entry = _entry(ledger)
    recommendation = evaluate(ledger, NOW)
    assert recommendation["new_arm_specification"] is None
    assert recommendation["exhaustion_not_decidable"] == [
        {
            "game": "r11l",
            "seed": 20260719,
            "window": 120,
            "source": entry["source"],
            "levels": 2,
            "stagnations_unredirected": 14,
            "reason": NOT_DECIDABLE_NO_WINDOW_ROWS,
        }
    ]
    assert recommendation["evidence"]["exhaustion_not_decidable"] == 1
    report = render_report(recommendation)
    assert "EXHAUSTION NOT DECIDABLE for 1 receipt(s)" in report
    assert "stagnations_unredirected=14 reason=no_window_rows_recorded" in report


def test_scenario_7033_c_rows_dropped_past_the_cap_are_listed_when_no_kept_row_decides(
    tmp_path: Path,
) -> None:
    """A receipt whose kept rows show no exhaustion but which dropped rows past the cap is
    listed with `window_rows_dropped_past_cap`; a kept row that IS exhausted still decides."""
    undecided = _applied_row(
        seed=1,
        redirects=[_redirect(arm, False) for arm in LEGACY_DEFAULT_ARMS],
        stag=70,
        arms_enabled=list(LEGACY_DEFAULT_ARMS),
        windows=_windows(64, level=2, arms_used=[ARM_DROP_GOAL_BIAS]),
        dropped=6,
    )
    decided = _applied_row(
        seed=2,
        redirects=[_redirect(arm, False) for arm in LEGACY_DEFAULT_ARMS],
        stag=70,
        arms_enabled=list(LEGACY_DEFAULT_ARMS),
        windows=_windows(64, level=2, arms_used=list(LEGACY_DEFAULT_ARMS)),
        dropped=6,
    )
    ledger, _ = _ingest(tmp_path, [undecided, decided])
    recommendation = evaluate(ledger, NOW)
    assert [
        (u["seed"], u["reason"], u["windows_dropped"])
        for u in recommendation["exhaustion_not_decidable"]
    ] == [(1, NOT_DECIDABLE_ROWS_DROPPED, 6)]
    cells = recommendation["new_arm_specification"]["cells"]
    assert [(c["seed"], c["level"], c["windows_dropped"]) for c in cells] == [(2, 2, 6)]


def test_scenario_7033_d_legacy_rows_alone_do_not_move_the_status(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7033-D: the 2026-09-05 ledger shape (four stagnating legacy rows,
    three arms fired each, 53 windows) reads `insufficient_evidence`, not a recommendation.
    The pooled trigger had moved the status on these same rows with five false cells."""
    rows = [
        _applied_row(
            seed=s, redirects=[_redirect(arm, False) for arm in LEGACY_DEFAULT_ARMS], stag=n
        )
        for s, n in ((1, 13), (2, 14), (3, 14), (4, 12))
    ]
    ledger, _ = _ingest(tmp_path, rows)
    recommendation = evaluate(ledger, NOW)
    assert recommendation["status"] == STATUS_INSUFFICIENT
    assert recommendation["new_arm_specification"] is None
    assert len(recommendation["exhaustion_not_decidable"]) == 4
    assert recommendation["evidence"]["stagnations_unredirected_total"] == 53
    assert STATUS_RECOMMENDATION not in render_report(recommendation)
