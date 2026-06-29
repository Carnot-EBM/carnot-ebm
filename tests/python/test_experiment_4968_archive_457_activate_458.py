"""Tests for REQ-CAPSTONE-4968 / SCENARIO-CAPSTONE-4968."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4968_archive_457_activate_458 as mod


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
        "a3_substrate_flag_resolved": True,
        "arc_first_win_wall_closed": True,
        "b3_window_nonzero": True,
        "banks_counted": {
            "audit_failure_reasons": [],
            "b1_banks_trustworthy": True,
            "bank_delta_counted": 0,
            "base_total_from_registry": 69,
            "candidate_banks": [
                {"game": "tr87", "new_levels_banked": 0, "source": "A1_LEVELUP"},
                {"game": "s5i5", "new_levels_banked": 0, "source": "A2_LEVELUP"},
            ],
            "computed_total": 69,
            "counted": [],
        },
        "capstone_ready": True,
        "heldout_first_win_rate": {
            "flag_resolved": True,
            "games_evaluated": 25,
            "honest_verdict": "complete_heldout_first_win_0.04_full25_final_flag_resolved",
            "rate": 0.04,
        },
        "honest_verdict": (
            "complete_capstone_v457_submission_ready_levels_69_heldout_0.04_"
            "package_ready_pivot_turnkey_7_1"
        ),
        "milestone_scorecard": {
            "a3_substrate_fix": {
                "decision": "a3_substrate_fix_holds_self_play_counted",
                "honest_verdict": "success_self_play_checkpoint_refreshed",
                "inference_substrate": "verifier_ensemble_against_cached_candidates",
                "offline_reproduced": True,
                "reproduced_levels": 2,
                "resolved": True,
                "summarizer_exit_code": 0,
                "target_game": "dc22",
                "true_live_recheck": "clean",
                "verifier_checkpoint_refreshed": True,
            },
            "b3_window_fix": {
                "decision": "b3_relaxed_mtime_window_maintained",
                "honest_verdict": "success_v457_stamping_backfilled_and_mtime_window_confirmed",
                "mtime_fallback_window": {
                    "compute_bound_count": 2,
                    "milestone": "2026.06.457",
                    "n_arms": 8,
                    "wall_minutes": 99.12,
                    "window_end": "2026-06-29T08:29:10Z",
                    "window_start": "2026-06-29T06:50:02Z",
                },
                "n_arms": 8,
                "nonzero": True,
                "research_conductor_modified": False,
                "wall_minutes": 99.12,
                "wiring_proposal_reconfirmed": True,
            },
            "banks": {
                "audit_failure_reasons": [],
                "b1_banks_trustworthy": True,
                "bank_delta_counted": 0,
                "base_total_from_registry": 69,
                "candidate_banks": [
                    {"game": "tr87", "new_levels_banked": 0, "source": "A1_LEVELUP"},
                    {"game": "s5i5", "new_levels_banked": 0, "source": "A2_LEVELUP"},
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
                "arxiv_ids_cited": [
                    "2508.16665",
                    "2508.10539",
                    "2502.11157",
                    "2605.18871",
                    "2504.16828",
                    "2502.01989",
                ],
                "b1_pivot_readiness_trustworthy": True,
                "d_pivot_executable_on_7_1": True,
                "d_pivot_turnkey": True,
                "decision": (
                    "post_6_30_distributional_energy_verifier_moat_experiment_"
                    "turnkey_7_1"
                ),
                "deliverable": "current ~0.05 agent + publishable FoVer paper",
                "do_not_queue": "representation_5",
                "extended_sota_backlog": ["2508.16665", "2508.10539", "2502.11157"],
                "moat_proven": False,
                "moat_proven_claimed": False,
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
                "ready_package_regression_ok": True,
                "submits": False,
            },
            "wall_closure": {
                "closed": True,
                "closure_verdict": "WALL_IS_HIDDEN_STATE",
                "did_reopen_in_v457": False,
                "do_not_queue": "representation_5",
                "representation_5_queued": False,
                "source": "standing_453_b1_trusted_closure",
                "trusted": True,
            },
        },
        "pivot_executable_on_7_1": True,
        "post_sprint_pivot": {
            "arxiv_id": "2605.18871",
            "arxiv_ids_cited": [
                "2508.16665",
                "2508.10539",
                "2502.11157",
                "2605.18871",
                "2504.16828",
                "2502.01989",
            ],
            "b1_pivot_readiness_trustworthy": True,
            "d_pivot_executable_on_7_1": True,
            "d_pivot_turnkey": True,
            "do_not_queue": "representation_5",
            "extended_sota_backlog": ["2508.16665", "2508.10539", "2502.11157"],
            "moat_proven": False,
            "pivot_executable_on_7_1": True,
            "pivot_turnkey": True,
            "sota_signal": "2605.18871 beats self-consistency on MuSR",
        },
        "reproducible_total_levels": 69,
        "submission_package_ready": {
            "frozen_stack_loads": True,
            "operator_only": True,
            "package_builds": True,
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
    active_milestone: str = "2026.06.458",
) -> None:
    if include_active:
        _write_text(root / "research-roadmap.yaml", f"milestone: {active_milestone}\n")
    if include_next:
        _write_text(root / "research-roadmap-next.yaml", "milestone: 2026.06.458\n")
    _write_text(root / mod.REGISTRY_REL_PATH, "schema_version: 1\nreproducible_total_levels: 69\n")
    _write_json(
        root / mod.RETRO_REL_PATH,
        {
            "experiments_completed": 0,
            "milestone": "2026.06.457",
            "summary": "Conductor false-zero retro gap; artifacts confirm .457 completed.",
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


def test_req_capstone_4968_spec_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4968: OpenSpec declares the .457/.458 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-4968") :]

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


def test_scenario_capstone_4968_complete_transition_uses_active_roadmap_without_next(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4968: active roadmap is enough after next was consumed."""

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
    assert calls[2] == [
        ".venv/bin/pytest",
        "tests/python/test_experiment_4968_archive_457_activate_458.py",
        "-q",
        "--no-cov",
    ]
    assert artifact["honest_verdict"] == (
        "complete_457_archived_458_activated_final_stretch_sprint_pivot_turnkey_recorded"
    )
    assert artifact["pretest_gate"]["ran"] is True
    assert artifact["pretest_gate"]["green"] is True
    assert artifact["transition"]["active_milestone_confirmed"] == "2026.06.458"
    assert artifact["transition"]["active_roadmap_path"] == "research-roadmap.yaml"
    assert artifact["transition"]["activation_state"] == "already_active_or_activated_458"
    assert artifact["transition_performed"] is True
    assert artifact["leaderboard_submission"] is False
    assert artifact["arc_first_win_wall_closed_hidden_state"] is True
    assert artifact["deliverable_locked_agent_plus_fover_paper"] is True
    assert artifact["v458_is_final_stretch_sprint_plus_pivot_turnkey"] is True
    assert artifact["reproducible_total_levels"] == 69

    close = artifact["close_state_457"]
    assert close["capstone"]["honest_verdict"] == (
        "complete_capstone_v457_submission_ready_levels_69_heldout_0.04_"
        "package_ready_pivot_turnkey_7_1"
    )
    assert close["a1_a2_no_banked"]["no_banked"] is True
    assert close["a1_a2_no_banked"]["candidate_banks"] == [
        {"game": "tr87", "new_levels_banked": 0, "source": "A1_LEVELUP"},
        {"game": "s5i5", "new_levels_banked": 0, "source": "A2_LEVELUP"},
    ]
    assert close["a1_a2_no_banked"]["fourth_consecutive_flat_milestone"] is True
    assert close["a1_a2_no_banked"]["deepen_well_dry_across_all_depth_regimes"] is True
    assert close["a3_self_play"]["target_game"] == "dc22"
    assert close["a3_self_play"]["verifier_checkpoint_refreshed"] is True
    assert close["a3_self_play"]["duration_too_short_flagged"] is False
    assert close["a4_heldout"]["heldout_first_win_rate"] == 0.04
    assert close["a4_heldout"]["flag_resolved"] is True
    assert close["a4_heldout"]["tautology_warn_only"] is True
    assert close["d_pivot"]["pivot_turnkey"] is True
    assert close["d_pivot"]["pivot_executable_on_7_1"] is True
    assert close["d_pivot"]["extended_sota_backlog"] == [
        "2508.16665",
        "2508.10539",
        "2502.11157",
    ]
    assert close["b2_package"]["peak_vram_gb"] == 15.146
    assert close["b2_package"]["peak_vram_lt_16"] is True
    assert close["b3_stamping"]["window_gate_relaxed"] is True
    assert close["b3_stamping"]["window_nonzero"] is True
    assert close["wall_closure"]["do_not_queue"] == ["representation_5"]
    assert close["concluded_levers_not_reopened"] == [
        "representation_5",
        "S0_oracle_distinct_structural_energy_program",
    ]
    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4968_blocked_missing_roadmaps_writes_deliverable(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4968-BLOCKED-PRECONDITION: neither roadmap parses."""

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
    assert artifact["close_state_457"]["reproducible_total_levels"] == 69
    assert artifact["arc_first_win_wall_closed_hidden_state"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4968_blocked_offline_arcade_skips_pretest(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4968-BLOCKED-PRECONDITION: offline arcade failure blocks."""

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


def test_scenario_capstone_4968_pretest_failure_blocks_after_preconditions(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4968: red pre-test gate is recorded, not hidden."""

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


def test_scenario_capstone_4968_missing_capstone_blocks_before_pretest(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4968-BLOCKED-PRECONDITION: missing close-state blocks."""

    _make_root(tmp_path, include_capstone=False)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
        started_s=2.0,
        now_s=2.25,
    )

    assert artifact["honest_verdict"] == "blocked_capstone_v457_missing"
    assert len(calls) == 2
    assert artifact["pretest_gate"]["ran"] is False
    assert artifact["transition_performed"] is False
    assert artifact["close_state_457"]["capstone"]["honest_verdict"] == ""
    assert artifact["arc_first_win_wall_closed_hidden_state"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4968_resource_blockers_and_bad_inputs(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4968: malformed resources fail closed without fabrication."""

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
        "capstone_v457": {"exists": True, "loadable": True},
    }
    roadmap_bad = {**ok, "active_roadmap_yaml": {"passed": False, "active_exists": True}}
    registry_missing = {**ok, "registry": {"exists": False, "loadable": False}}
    registry_bad = {**ok, "registry": {"exists": True, "loadable": False}}
    capstone_bad = {**ok, "capstone_v457": {"exists": True, "loadable": False}}

    assert mod.precondition_blocker(roadmap_bad) == "blocked_roadmap_yaml_unparseable"
    assert mod.precondition_blocker(registry_missing) == "blocked_arc_solve_registry_missing"
    assert mod.precondition_blocker(registry_bad) == "blocked_arc_solve_registry_unloadable"
    assert mod.precondition_blocker(capstone_bad) == "blocked_capstone_v457_unloadable"

    _make_root(tmp_path, include_next=True)
    cited_paths = [row["path"] for row in mod.cited_upstream_artifacts(tmp_path)]
    assert "research-roadmap-next.yaml" in cited_paths
