"""Spec: REQ-ARC-WMTE-7030, REQ-ARC-WMTE-7031, REQ-ARC-WMTE-7032 and their scenarios.

The trajectory supervisor says which arms its run could fire and what it saw at each window
it could not answer; the refinement tool reads both, and keeps shadow receipts as controls.

MEASURED 2026-09-05 (population: the 14 receipts in ops/arc_supervisor_refinement_ledger.json
and the 14 artifacts in results/arc_leaderboard_eval_runs/):

- All 64 unredirected windows sit in 5 eval receipts where the three default-on arms all
  fired. Only ONE of those runs had the env-gated tool rung on, so a trigger that waits for
  `set(ARM_ORDER)` reported one cell and hid four (53 of 64 windows). REQ-7030.
- No receipt says WHY the table ran dry (attempt cap? diversity already on?). REQ-7031.
- One shadow receipt (r11l, seed 20260719, window 120) leveled up at action 813 with nothing
  applied, and its would-have rows at 120/240/360 carry the same 765/645/525 actions-to-level-up
  that four applied runs booked as `helped`. A control now exists; the tool ignored it. REQ-7032.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.agentic.arc_supervisor_refinement import (
    LEGACY_DEFAULT_ARMS,
    MIN_FIRED_PER_ARM,
    STATUS_RECOMMENDATION,
    empty_ledger,
    enabled_arms_for_entry,
    evaluate,
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


def _redirect(arm: str, resolved: bool, level: int = 0, a2l: int | None = None) -> dict:
    return {
        "arm": arm,
        "action_index": 120,
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
        receipt["unredirected_windows_dropped"] = 0
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


def _exhausted_windows(n: int, **state) -> list[dict]:
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
    return [{"action_index": 1490 + 120 * i, **base} for i in range(n)]


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


def test_scenario_7030_c_legacy_row_with_three_default_arms_is_an_exhausted_cell(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7030-C: the 2026-09-05 shape. A row with no `arms_enabled` that
    fired the three default arms and still stagnated IS a new-arm cell; reading ARM_ORDER
    (which holds the default-off tool rung) hid four of five such cells."""
    row = _applied_row(
        redirects=[_redirect(arm, True, a2l=700) for arm in LEGACY_DEFAULT_ARMS],
        stag=14,
    )
    ledger, _ = _ingest(tmp_path, [row])
    entry = next(iter(ledger["entries"].values()))
    assert entry["arms_enabled"] is None
    enabled, source = enabled_arms_for_entry(entry)
    assert enabled == set(LEGACY_DEFAULT_ARMS)
    assert source == "legacy_default"
    spec = evaluate(ledger, NOW)["new_arm_specification"]
    assert spec is not None
    assert len(spec["cells"]) == 1
    cell = spec["cells"][0]
    assert cell["arms_enabled_source"] == "legacy_default"
    assert cell["stagnations_unredirected"] == 14
    assert "arms_enabled" in spec["trigger"]


def test_scenario_7030_c_a_declared_four_arm_run_that_fired_three_is_not_exhausted(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7030-C: a run that declares the tool rung enabled but never fired
    it still had a rung left, so it is not a cell. The declared set wins over the legacy default."""
    declared_four = _applied_row(
        seed=1,
        redirects=[_redirect(arm, False) for arm in LEGACY_DEFAULT_ARMS],
        stag=5,
        arms_enabled=list(ARM_ORDER),
    )
    declared_three = _applied_row(
        seed=2,
        redirects=[_redirect(arm, False) for arm in LEGACY_DEFAULT_ARMS],
        stag=5,
        arms_enabled=list(LEGACY_DEFAULT_ARMS),
    )
    ledger, _ = _ingest(tmp_path, [declared_four, declared_three])
    spec = evaluate(ledger, NOW)["new_arm_specification"]
    assert spec is not None
    assert [(c["seed"], c["arms_enabled_source"]) for c in spec["cells"]] == [(2, "receipt")]


def test_scenario_7030_c_a_fired_arm_is_unioned_into_a_declared_set(tmp_path: Path) -> None:
    """A receipt that declares three arms but fired the tool rung anyway (a late env flip
    on an old receipt) counts the tool rung as enabled, so the fired set is exhausted only
    when every one of the four is present."""
    row = _applied_row(
        redirects=[_redirect(arm, False) for arm in ARM_ORDER],
        stag=3,
        arms_enabled=list(LEGACY_DEFAULT_ARMS),
    )
    ledger, _ = _ingest(tmp_path, [row])
    enabled, _ = enabled_arms_for_entry(next(iter(ledger["entries"].values())))
    assert enabled == set(ARM_ORDER)
    assert evaluate(ledger, NOW)["new_arm_specification"] is not None


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


def test_scenario_7031_c_the_cell_summarises_the_exhaustion_states(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7031-C: the ledger keeps the rows and the new-arm cell counts them
    per flag; the report prints the counts."""
    windows = _exhausted_windows(9) + _exhausted_windows(
        2,
        level=0,
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
    entry = next(iter(ledger["entries"].values()))
    assert len(entry["unredirected_windows"]) == 11
    assert entry["unredirected_windows_dropped"] == 0
    recommendation = evaluate(ledger, NOW)
    states = recommendation["new_arm_specification"]["cells"][0]["exhaustion_states"]
    assert states == {
        "windows_recorded": 11,
        "windows_dropped": 0,
        "levels": [0, 2],
        "arms_used_sets": sorted(
            {
                ",".join([ARM_DROP_GOAL_BIAS, ARM_ALLOW_REINDUCTION]),
                ",".join(LEGACY_DEFAULT_ARMS),
            }
        ),
        "goal_bias_installed": 0,
        "induced": 11,
        "attempt_cap_reached": 9,
        "evidence_floor_met": 11,
        "diversity_active": 11,
    }
    report = render_report(recommendation)
    assert "states: windows=11 levels=[0, 2] attempt_cap_reached=9" in report


def test_scenario_7031_c_a_legacy_row_reads_not_recorded(tmp_path: Path) -> None:
    """A row from before the field must not read as "every flag False"."""
    row = _applied_row(redirects=[_redirect(arm, False) for arm in LEGACY_DEFAULT_ARMS], stag=14)
    ledger, _ = _ingest(tmp_path, [row])
    entry = next(iter(ledger["entries"].values()))
    assert entry["unredirected_windows"] == []
    assert exhaustion_summary(entry) == "not_recorded"
    recommendation = evaluate(ledger, NOW)
    assert (
        recommendation["new_arm_specification"]["cells"][0]["exhaustion_states"] == "not_recorded"
    )
    assert "states: not_recorded" in render_report(recommendation)


def test_scenario_7031_c_the_ledger_copy_is_bounded_too(tmp_path: Path) -> None:
    """An oversized producer list cannot bloat the ledger past the receipt's own cap."""
    row = _applied_row(
        redirects=[_redirect(arm, False) for arm in LEGACY_DEFAULT_ARMS],
        stag=MAX_UNREDIRECTED_WINDOWS + 10,
        windows=_exhausted_windows(MAX_UNREDIRECTED_WINDOWS + 10),
    )
    ledger, _ = _ingest(tmp_path, [row])
    entry = next(iter(ledger["entries"].values()))
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


def test_scenario_7032_status_is_recommendation_when_only_legacy_cells_exist(
    tmp_path: Path,
) -> None:
    """The widened trigger alone moves the ledger's status; the human sees every cell."""
    rows = [
        _applied_row(
            seed=s, redirects=[_redirect(arm, False) for arm in LEGACY_DEFAULT_ARMS], stag=n
        )
        for s, n in ((1, 13), (2, 14), (3, 14), (4, 12))
    ]
    ledger, _ = _ingest(tmp_path, rows)
    recommendation = evaluate(ledger, NOW)
    assert recommendation["status"] == STATUS_RECOMMENDATION
    assert len(recommendation["new_arm_specification"]["cells"]) == 4
    assert recommendation["evidence"]["stagnations_unredirected_total"] == 53
