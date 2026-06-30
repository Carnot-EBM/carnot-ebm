"""Tests for REQ-CAPSTONE-5001 / SCENARIO-CAPSTONE-5001."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5001_archive_460_activate_461 as mod


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
    backlog = [
        "2504.13134",
        "2605.10158",
        "2605.18871",
        "2504.16828",
        "2502.01989",
        "2508.16665",
        "2508.10539",
        "2502.11157",
        "2504.01005",
        "2504.00891",
        "2509.24460",
        "2510.14913",
        "2603.04304",
    ]
    pivot = {
        "arxiv_id": "2605.18871",
        "arxiv_ids_cited": backlog,
        "b1_pivot_readiness_trustworthy": True,
        "claim_status": "readiness_only_not_moat_proven",
        "d_pivot_executable_on_7_1": True,
        "d_pivot_turnkey": True,
        "decision": "post_6_30_distributional_energy_verifier_moat_experiment_turnkey_7_1",
        "deliverable": "current ~0.05 agent + publishable FoVer paper",
        "do_not_queue": "representation_5_or_concluded_energy_as_arc_program",
        "extended_sota_backlog": backlog,
        "extended_sota_backlog_count": 13,
        "moat_proven": False,
        "moat_proven_claimed": False,
        "new_sota_backlog_ids": ["2504.13134", "2605.10158"],
        "pivot_executable_on_7_1": True,
        "pivot_turnkey": True,
        "runs_after": "2026-06-30_sprint_retirement",
        "sota_signal": "2605.18871 beats self-consistency on MuSR",
        "validation_gate": {"claimed_met": False, "verifier_is_oracle_required_value": False},
    }
    return {
        "a3_substrate_flag_resolved": True,
        "arc_first_win_wall_closed": True,
        "b3_window_nonzero": True,
        "banks_counted": {
            "audit_failure_reasons": [],
            "b1_banks_trustworthy": True,
            "bank_delta_counted": 0,
            "base_total_from_registry": 69,
            "candidate_banks": [
                {"game": "sc25", "new_levels_banked": 0, "source": "A1_LEVELUP"},
                {"game": "sk48", "new_levels_banked": 0, "source": "A2_LEVELUP"},
            ],
            "computed_total": 69,
            "counted": [],
        },
        "capstone_ready": True,
        "headline": (
            ".460 deadline-bound submission readiness: locked ~0.05 agent + FoVer paper "
            "ready for 6/30 at 69 reproducible levels; held-out first-win 0.04 with "
            "flag_resolved=True; package ready; post-6/30 verifier-moat pivot turnkey; "
            "13-paper SOTA backlog; ARC wall remains WALL_IS_HIDDEN_STATE closed."
        ),
        "heldout_first_win_rate": {
            "flag_resolved": True,
            "games_evaluated": 25,
            "honest_verdict": "complete_heldout_first_win_0.04_full25_final_flag_resolved",
            "rate": 0.04,
            "status": "full25_go_no_go",
        },
        "honest_verdict": (
            "complete_capstone_v460_submission_ready_levels_69_heldout_0.04_"
            "package_ready_pivot_turnkey_7_1"
        ),
        "milestone_scorecard": {
            "a3_substrate_fix": {
                "flag_resolved": True,
                "honest_verdict": "success_self_play_checkpoint_refreshed",
                "reproduced_levels": 3,
                "target_game": "ft09",
                "true_live_recheck": "clean",
                "verifier_checkpoint_refreshed": True,
            },
            "b3_window_fix": {
                "decision": "b3_relaxed_mtime_window_maintained",
                "honest_verdict": "success_v460_stamping_backfilled_and_mtime_window_confirmed",
                "mtime_fallback_window": {
                    "milestone": "2026.06.460",
                    "n_arms": 8,
                    "wall_minutes": 152.78,
                },
                "nonzero": True,
                "research_conductor_modified": False,
            },
            "banks": {
                "candidate_banks": [
                    {"game": "sc25", "new_levels_banked": 0, "source": "A1_LEVELUP"},
                    {"game": "sk48", "new_levels_banked": 0, "source": "A2_LEVELUP"},
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
            "post_sprint_pivot": pivot,
            "submission_package": {
                "decision": "package_ready_operator_only",
                "frozen_stack_loads": True,
                "honest_verdict": "success_submission_package_ready_final_pre_deadline",
                "operator_only": True,
                "package_builds": True,
                "peak_vram_gb": 15.146,
                "ready": True,
                "ready_package_regression_ok": True,
                "submits": False,
            },
            "wall_closure": {
                "closed": True,
                "closure_verdict": "WALL_IS_HIDDEN_STATE",
                "did_reopen_energy_as_arc_program": False,
                "do_not_queue": "representation_5_or_concluded_energy_as_arc_program",
                "energy_as_arc_program": "S0_CONCLUDED_2026_06_26",
                "representation_5_queued": False,
                "trusted": True,
            },
        },
        "pivot_executable_on_7_1": True,
        "post_sprint_pivot": pivot,
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
    active_milestone: str = "2026.06.461",
) -> None:
    roadmap_text = (
        f"milestone: {active_milestone}\n"
        "note: 'ARC sprint RETIRED; PHASE D majority lever; paper_ready=true; "
        "operator directive 2026-06-30.'\n"
    )
    if include_active:
        _write_text(root / "research-roadmap.yaml", roadmap_text)
    if include_next:
        _write_text(root / "research-roadmap-next.yaml", "milestone: 2026.06.461\n")
    _write_text(root / mod.REGISTRY_REL_PATH, "schema_version: 1\nreproducible_total_levels: 69\n")
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


def test_req_capstone_5001_spec_declares_transition_contract() -> None:
    """REQ-CAPSTONE-5001: OpenSpec declares the .460/.461 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5001") :]

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
    assert "PHASE D is unlocked as the new majority lever" in section
    assert mod.PRETEST_COMMAND == [
        ".venv/bin/pytest",
        "tests/python/test_experiment_5001_archive_460_activate_461.py",
        "-q",
        "--no-cov",
    ]


def test_scenario_capstone_5001_complete_transition_records_phase_d(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5001: active .461 roadmap is enough after next was consumed."""

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
    assert calls[2] == mod.PRETEST_COMMAND
    assert artifact["honest_verdict"] == "complete_460_archived_461_activated_phase_d_unlocked"
    assert artifact["pretest_gate"]["ran"] is True
    assert artifact["pretest_gate"]["green"] is True
    assert artifact["transition"]["active_milestone_confirmed"] == "2026.06.461"
    assert artifact["transition"]["active_roadmap_path"] == "research-roadmap.yaml"
    assert artifact["transition"]["activation_state"] == "already_active_or_activated_461"
    assert artifact["transition_performed"] is True
    assert artifact["leaderboard_submission"] is False
    assert artifact["arc_sprint_retired"] is True
    assert artifact["phase_d_unlocked_majority_lever"] is True
    assert artifact["arc_deliverable_locked"] is True
    assert artifact["reproducible_total_levels"] == 69

    close = artifact["close_state_460"]
    assert close["capstone"]["honest_verdict"] == (
        "complete_capstone_v460_submission_ready_levels_69_heldout_0.04_"
        "package_ready_pivot_turnkey_7_1"
    )
    assert close["a1_a2_no_banked"]["no_banked"] is True
    assert close["a1_a2_no_banked"]["candidate_banks"] == [
        {"game": "sc25", "new_levels_banked": 0, "source": "A1_LEVELUP"},
        {"game": "sk48", "new_levels_banked": 0, "source": "A2_LEVELUP"},
    ]
    assert close["a1_a2_no_banked"]["sixth_consecutive_flat_milestone"] is True
    assert close["a1_a2_no_banked"]["deepen_well_dry_across_all_depth_regimes"] is True
    assert close["a4_heldout"]["heldout_first_win_rate"] == 0.04
    assert close["a4_heldout"]["games_evaluated"] == 25
    assert close["a4_heldout"]["flag_resolved"] is True
    assert close["b2_package"]["ready"] is True
    assert close["b2_package"]["peak_vram_gb"] == 15.146
    assert close["d_pivot"]["pivot_turnkey"] is True
    assert close["d_pivot"]["pivot_executable_on_7_1"] is True
    assert close["d_pivot"]["extended_sota_backlog_count"] == 13
    assert close["d_pivot"]["new_sota_backlog_ids"] == ["2504.13134", "2605.10158"]
    assert close["arc_sprint_retired"]["retired"] is True
    assert close["arc_sprint_retired"]["retired_date"] == "2026-06-30"
    assert close["arc_deliverable"]["locked"] is True
    assert close["arc_deliverable"]["paper_ready"] is True
    assert close["phase_d"]["unlocked"] is True
    assert close["phase_d"]["majority_lever"] is True
    assert close["wall_closure"]["closure_verdict"] == "WALL_IS_HIDDEN_STATE"
    assert close["wall_closure"]["do_not_queue"] == [
        "representation_5",
        "concluded_energy_as_arc_program",
    ]
    assert close["energy_as_arc_program_concluded"] is True
    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5001_blocked_missing_roadmaps_writes_deliverable(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5001-BLOCKED-PRECONDITION: neither roadmap parses."""

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
    assert artifact["transition"]["active_milestone_confirmed"] == "unknown"
    assert artifact["transition_performed"] is False
    assert artifact["close_state_460"]["reproducible_total_levels"] == 69
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5001_blocked_offline_arcade_skips_pretest(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5001-BLOCKED-PRECONDITION: offline arcade failure blocks."""

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


def test_scenario_capstone_5001_pretest_failure_blocks_after_preconditions(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5001: red pre-test gate is recorded, not hidden."""

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


def test_scenario_capstone_5001_missing_capstone_blocks_before_pretest(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5001-BLOCKED-PRECONDITION: missing close-state blocks."""

    _make_root(tmp_path, include_capstone=False)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
        started_s=2.0,
        now_s=2.25,
    )

    assert artifact["honest_verdict"] == "blocked_capstone_v460_missing"
    assert len(calls) == 2
    assert artifact["pretest_gate"]["ran"] is False
    assert artifact["transition_performed"] is False
    assert artifact["close_state_460"]["capstone"]["honest_verdict"] == ""
    assert artifact["phase_d_unlocked_majority_lever"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5001_resource_blockers_and_bad_inputs(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5001: malformed resources fail closed without fabrication."""

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
        "capstone_v460": {"exists": True, "loadable": True},
    }
    roadmap_bad = {**ok, "active_roadmap_yaml": {"passed": False, "active_exists": True}}
    registry_missing = {**ok, "registry": {"exists": False, "loadable": False}}
    registry_bad = {**ok, "registry": {"exists": True, "loadable": False}}
    capstone_bad = {**ok, "capstone_v460": {"exists": True, "loadable": False}}

    assert mod.precondition_blocker(roadmap_bad) == "blocked_roadmap_yaml_unparseable"
    assert mod.precondition_blocker(registry_missing) == "blocked_arc_solve_registry_missing"
    assert mod.precondition_blocker(registry_bad) == "blocked_arc_solve_registry_unloadable"
    assert mod.precondition_blocker(capstone_bad) == "blocked_capstone_v460_unloadable"

    _make_root(tmp_path, include_next=True)
    cited_paths = [row["path"] for row in mod.cited_upstream_artifacts(tmp_path)]
    assert "research-roadmap-next.yaml" in cited_paths
