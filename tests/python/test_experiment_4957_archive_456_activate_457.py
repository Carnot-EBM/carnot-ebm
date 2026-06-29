"""Tests for REQ-CAPSTONE-4957 / SCENARIO-CAPSTONE-4957."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4957_archive_456_activate_457 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _capstone_artifact() -> JsonDict:
    return {
        "arc_first_win_wall_closed": True,
        "b3_window_nonzero": True,
        "capstone_ready": True,
        "headline": (
            ".456 final submission readiness: locked ~0.05 agent + FoVer paper ready "
            "for 6/30 at 69 reproducible levels."
        ),
        "heldout_first_win_rate": {
            "flag_resolved": True,
            "games_evaluated": 25,
            "honest_verdict": "complete_heldout_first_win_0.04_full25_final_flag_resolved",
            "rate": 0.04,
        },
        "honest_verdict": (
            "complete_capstone_v456_submission_ready_levels_69_heldout_0.04_"
            "package_ready_pivot_turnkey_7_1"
        ),
        "milestone_scorecard": {
            "a3_substrate": {
                "duration_s": 10.198359,
                "flag_resolved": True,
                "honest_verdict": "success_self_play_checkpoint_refreshed",
                "reproduction_gate": {"game": "lp85", "reproduced": True},
                "resolved": True,
                "status": "substrate_fix_counted",
                "target_game": "lp85",
                "verifier_checkpoint_refreshed": True,
            },
            "b3_window": {
                "honest_verdict": "success_v456_stamping_backfilled_and_mtime_window_confirmed",
                "mtime_fallback_window": {
                    "compute_bound_count": 4,
                    "n_arms": 8,
                    "wall_minutes": 100.2,
                    "window_end": "2026-06-29T00:49:13Z",
                    "window_start": "2026-06-28T23:09:01Z",
                },
                "research_conductor_modified": False,
                "window_gate_relaxed": True,
                "window_nonzero": True,
            },
            "banks": {
                "audit_failure_reasons": [],
                "b1_banks_trustworthy": True,
                "bank_delta_counted": 0,
                "base_total_from_registry": 69,
                "candidate_banks": [
                    {"game": "ar25", "new_levels_banked": 0, "source": "A1_LEVELUP"},
                    {"game": "vc33", "new_levels_banked": 0, "source": "A2_LEVELUP"},
                ],
                "computed_total": 69,
                "counted": [],
            },
            "deliverable": "locked ~0.05 first-win agent + publishable FoVer paper",
            "heldout_go_no_go": {
                "flag_resolved": True,
                "games_evaluated": 25,
                "honest_verdict": "complete_heldout_first_win_0.04_full25_final_flag_resolved",
                "rate": 0.04,
            },
            "post_sprint_pivot": {
                "arxiv_id": "2605.18871",
                "arxiv_ids_cited": ["2605.18871", "2504.16828", "2502.01989"],
                "b1_pivot_readiness_trustworthy": True,
                "decision": "post_6_30_distributional_energy_verifier_moat_experiment_turnkey_7_1",
                "deliverable": "current ~0.05 agent + publishable FoVer paper",
                "do_not_queue": "representation_5",
                "moat_proven": False,
                "pivot_executable_on_7_1": True,
                "pivot_turnkey": True,
                "sota_signal": "2605.18871 beats self-consistency on MuSR",
            },
            "submission_package": {
                "decision": "package_ready_operator_only",
                "frozen_stack_loads": True,
                "honest_verdict": "success_submission_package_ready_final_pre_deadline",
                "operator_only": True,
                "package_builds": True,
                "peak_vram_gb": 15.146,
                "ready": True,
                "submits": False,
            },
            "wall_closure": {
                "closed": True,
                "closure_verdict": "WALL_IS_HIDDEN_STATE",
                "did_reopen_in_v456": False,
                "do_not_queue": "representation_5",
                "trusted": True,
            },
        },
        "pivot_executable_on_7_1": True,
        "post_sprint_pivot": {
            "arxiv_id": "2605.18871",
            "deliverable": "current ~0.05 agent + publishable FoVer paper",
            "do_not_queue": "representation_5",
            "pivot_executable_on_7_1": True,
            "pivot_turnkey": True,
            "sota_signal": "2605.18871 beats self-consistency on MuSR",
        },
        "reproducible_total_levels": 69,
        "submission_package_ready": {
            "frozen_stack_loads": True,
            "operator_only": True,
            "peak_vram_gb": 15.146,
            "ready": True,
            "submits": False,
        },
    }


def _make_root(
    root: Path,
    *,
    include_active: bool = True,
    include_next: bool = False,
    include_capstone: bool = True,
    active_milestone: str = "2026.06.457",
) -> None:
    if include_active:
        _write_text(root / "research-roadmap.yaml", f"milestone: {active_milestone}\n")
    if include_next:
        _write_text(root / "research-roadmap-next.yaml", "milestone: 2026.06.457\n")
    _write_text(root / mod.REGISTRY_REL_PATH, "schema_version: 1\nreproducible_total_levels: 69\n")
    _write_json(
        root / mod.RETRO_REL_PATH,
        {
            "experiments_completed": 0,
            "milestone": "2026.06.456",
            "summary": "Conductor timing detector showed the recurring false-zero gap.",
        },
    )
    if include_capstone:
        _write_json(root / mod.CAPSTONE_REL_PATH, _capstone_artifact())


def _runner(calls: list[list[str]], *, roadmap_ok: bool, offline_ok: bool, pretest_ok: bool):
    def run(command: list[str], _root: Path) -> mod.CommandResult:
        calls.append(command)
        command_text = " ".join(command)
        if "research-roadmap.yaml" in command_text:
            return mod.CommandResult(command, 0 if roadmap_ok else 1, "ok active\n", "roadmap")
        if "offline_arcade" in command_text:
            return mod.CommandResult(command, 0 if offline_ok else 1, "", "arcade")
        return mod.CommandResult(command, 0 if pretest_ok else 1, "passed", "failed")

    return run


def test_req_capstone_4957_spec_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4957: OpenSpec declares the .456/.457 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-4957") :]

    assert mod._do_not_queue(["representation_5"]) == ["representation_5"]
    for ref in mod.SPEC_REFS:
        assert ref in section
    assert str(mod.OUTPUT_REL_PATH) in section
    for field in mod.REQUIRED_FIELDS:
        assert field in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section
    assert "research-roadmap.yaml` first" in section
    assert "research-roadmap-next.yaml` SHALL NOT block" in section


def test_scenario_capstone_4957_complete_transition_uses_active_roadmap_without_next(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4957: active roadmap is enough after next was consumed."""

    _make_root(tmp_path, include_active=True, include_next=False)
    calls: list[list[str]] = []
    exit_code = mod.main(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
    )
    artifact = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert exit_code == 0
    assert len(calls) == 3
    assert "p if os.path.exists(p) else q" in " ".join(calls[0])
    assert artifact["honest_verdict"] == (
        "complete_456_archived_457_activated_final_sprint_day_pivot_turnkey_recorded"
    )
    assert artifact["pretest_gate"]["ran"] is True
    assert artifact["pretest_gate"]["green"] is True
    assert artifact["transition"]["active_milestone_confirmed"] == "2026.06.457"
    assert artifact["transition"]["active_roadmap_path"] == "research-roadmap.yaml"
    assert artifact["transition"]["activation_state"] == "already_active_or_activated_457"
    assert artifact["transition_performed"] is True
    assert artifact["leaderboard_submission"] is False
    assert artifact["arc_first_win_wall_closed_hidden_state"] is True
    assert artifact["deliverable_locked_agent_plus_fover_paper"] is True
    assert artifact["v457_is_final_sprint_day_plus_pivot_turnkey"] is True
    assert artifact["reproducible_total_levels"] == 69
    assert artifact["close_state_456"]["capstone"]["honest_verdict"] == (
        "complete_capstone_v456_submission_ready_levels_69_heldout_0.04_"
        "package_ready_pivot_turnkey_7_1"
    )
    assert artifact["close_state_456"]["a1_a2_no_banked"]["no_banked"] is True
    assert artifact["close_state_456"]["a1_a2_no_banked"]["candidate_banks"] == [
        {"game": "ar25", "new_levels_banked": 0, "source": "A1_LEVELUP"},
        {"game": "vc33", "new_levels_banked": 0, "source": "A2_LEVELUP"},
    ]
    assert artifact["close_state_456"]["a1_a2_no_banked"][
        "third_consecutive_flat_milestone"
    ] is True
    assert artifact["close_state_456"]["a3_self_play"]["verifier_checkpoint_refreshed"] is True
    assert artifact["close_state_456"]["a3_self_play"]["duration_too_short_flag_resolved"] is True
    assert artifact["close_state_456"]["a4_heldout"]["heldout_first_win_rate"] == 0.04
    assert artifact["close_state_456"]["a4_heldout"]["tautology_warn_only"] is True
    assert artifact["close_state_456"]["d_pivot"]["pivot_turnkey"] is True
    assert artifact["close_state_456"]["d_pivot"]["pivot_executable_on_7_1"] is True
    assert artifact["close_state_456"]["d_pivot"]["arxiv_id"] == "2605.18871"
    assert artifact["close_state_456"]["b2_package"]["peak_vram_gb"] == 15.146
    assert artifact["close_state_456"]["b2_package"]["peak_vram_lt_16"] is True
    assert artifact["close_state_456"]["b3_stamping"]["window_gate_relaxed"] is True
    assert artifact["close_state_456"]["b3_stamping"]["window_nonzero"] is True
    assert artifact["close_state_456"]["wall_closure"]["do_not_queue"] == ["representation_5"]
    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4957_blocked_missing_roadmaps_writes_deliverable(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4957-BLOCKED-PRECONDITION: neither roadmap parses."""

    _make_root(tmp_path, include_active=False, include_next=False)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=False, offline_ok=True, pretest_ok=True),
        started_s=10.0,
        now_s=10.25,
    )

    assert artifact["honest_verdict"] == "blocked_roadmap_yaml_missing"
    assert len(calls) == 2
    assert artifact["pretest_gate"] == {
        "green": False,
        "ran": False,
        "reason": "skipped_after_precondition_failure",
    }
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["transition_performed"] is False
    assert artifact["close_state_456"]["reproducible_total_levels"] == 69
    assert artifact["arc_first_win_wall_closed_hidden_state"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4957_blocked_offline_arcade_skips_pretest(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4957-BLOCKED-PRECONDITION: offline arcade failure blocks."""

    _make_root(tmp_path)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=False, pretest_ok=True),
        started_s=4.0,
        now_s=4.5,
    )

    assert artifact["honest_verdict"] == "blocked_offline_arcade_unavailable"
    assert len(calls) == 2
    assert artifact["pretest_gate"]["ran"] is False
    assert artifact["preconditions_checked"]["offline_arcade"]["passed"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4957_pretest_failure_blocks_after_preconditions(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4957: red pre-test gate is recorded, not hidden."""

    _make_root(tmp_path)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=True, pretest_ok=False),
        started_s=4.0,
        now_s=4.5,
    )

    assert artifact["honest_verdict"] == "blocked_pretest_gate_failed"
    assert len(calls) == 3
    assert artifact["pretest_gate"]["ran"] is True
    assert artifact["pretest_gate"]["green"] is False
    assert artifact["transition_performed"] is False
    assert artifact["poison_test_resolved"] == {"quarantined": False, "test": "", "reason": ""}
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4957_missing_capstone_blocks_before_pretest(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4957-BLOCKED-PRECONDITION: missing close-state blocks."""

    _make_root(tmp_path, include_capstone=False)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
        started_s=2.0,
        now_s=2.25,
    )

    assert artifact["honest_verdict"] == "blocked_capstone_v456_missing"
    assert len(calls) == 2
    assert artifact["pretest_gate"]["ran"] is False
    assert artifact["transition_performed"] is False
    assert artifact["close_state_456"]["capstone"]["honest_verdict"] == ""
    assert artifact["arc_first_win_wall_closed_hidden_state"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4957_resource_blockers_and_bad_inputs(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4957: malformed resources fail closed without fabrication."""

    bad_json = tmp_path / "bad.json"
    bad_yaml = tmp_path / "bad.yaml"
    bad_json.write_text("{", encoding="utf-8")
    bad_yaml.write_text("bad: [", encoding="utf-8")

    assert mod._read_json_object_safe(bad_json) == {}
    assert mod._read_yaml_object_safe(bad_yaml) == {}

    ok = {
        "active_roadmap_yaml": {"passed": True, "active_exists": True, "next_exists": False},
        "offline_arcade": {"passed": True},
        "registry": {"exists": True, "loadable": True},
        "capstone_v456": {"exists": True, "loadable": True},
    }
    roadmap_bad = {**ok, "active_roadmap_yaml": {"passed": False, "active_exists": True}}
    registry_missing = {**ok, "registry": {"exists": False, "loadable": False}}
    registry_bad = {**ok, "registry": {"exists": True, "loadable": False}}
    capstone_bad = {**ok, "capstone_v456": {"exists": True, "loadable": False}}

    assert mod.precondition_blocker(roadmap_bad) == "blocked_roadmap_yaml_unparseable"
    assert mod.precondition_blocker(registry_missing) == "blocked_arc_solve_registry_missing"
    assert mod.precondition_blocker(registry_bad) == "blocked_arc_solve_registry_unloadable"
    assert mod.precondition_blocker(capstone_bad) == "blocked_capstone_v456_unloadable"

    _make_root(tmp_path, include_next=True)
    cited_paths = [row["path"] for row in mod.cited_upstream_artifacts(tmp_path)]
    assert "research-roadmap-next.yaml" in cited_paths
