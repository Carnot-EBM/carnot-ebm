"""Tests for REQ-CAPSTONE-4946 / SCENARIO-CAPSTONE-4946."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4946_archive_455_activate_456 as mod


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
        "headline": (
            ".455 final submission readiness: locked ~0.05 agent + FoVer paper ready "
            "for 6/30 at 69 reproducible levels."
        ),
        "heldout_first_win_rate": {
            "flag_resolved": True,
            "games_evaluated": 25,
            "honest_verdict": "complete_heldout_first_win_0.04_full25_final_flag_resolved",
            "rate": 0.04,
        },
        "honest_verdict": (
            "complete_capstone_v455_submission_ready_levels_69_heldout_0.04_"
            "package_ready_pivot_executable_7_1"
        ),
        "milestone_scorecard": {
            "banks": {
                "audit_failure_reasons": [],
                "b1_banks_trustworthy": True,
                "bank_delta_counted": 0,
                "base_total_from_registry": 69,
                "candidate_banks": [
                    {"game": "lf52", "new_levels_banked": 0, "source": "A1_LEVELUP"},
                    {"game": "sb26", "new_levels_banked": 0, "source": "A2_LEVELUP"},
                ],
                "computed_total": 69,
                "counted": [],
            },
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
                "decision": "post_6_30_distributional_energy_verifier_moat_experiment_executable_7_1",
                "deliverable": "current ~0.05 agent + publishable FoVer paper",
                "do_not_queue": "representation_5",
                "executable_date": "2026-07-01",
                "moat_proven": False,
                "pivot_executable_on_7_1": True,
                "sota_signal": "2605.18871 beats self-consistency on MuSR",
            },
            "reserved_lanes": {
                "b3_stamping": {
                    "decision": "reserved_lane_blocked_insufficient_v455_mtime_window",
                    "honest_verdict": "blocked_insufficient_v455_mtime_window",
                    "mtime_fallback_window": {
                        "n_arms": 8,
                        "compute_bound_count": 3,
                        "wall_minutes": 112.46,
                    },
                    "research_conductor_modified": False,
                }
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
                "did_reopen_in_v455": False,
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
            "sota_signal": "2605.18871 beats self-consistency on MuSR",
        },
        "reproducible_total_levels": 69,
        "skipped_flagged_adversarial": [
            {
                "experiment_id": 4938,
                "reason": "live_critical_recheck",
                "source": "A3_SELF_PLAY",
                "true_live_recheck": "critical",
            }
        ],
        "submission_package_ready": {
            "operator_only": True,
            "peak_vram_gb": 15.146,
            "ready": True,
            "submits": False,
        },
    }


def _self_play_artifact() -> JsonDict:
    return {
        "duration_s": 0.0001,
        "experiment": "experiment_4938_self_play_verifier_checkpoint",
        "flagged_adversarial": True,
        "honest_verdict": "success_self_play_checkpoint_refreshed",
        "inference_substrate": "live_llm_inference",
        "corrigendum_pending": [
            {
                "detail": (
                    "duration_s=0.0001 but artifact references compute-bound markers; "
                    "real model takes >=60.0s minimum."
                ),
                "kind": "DURATION_TOO_SHORT",
                "severity": "critical",
            }
        ],
    }


def _stamping_artifact() -> JsonDict:
    return {
        "experiment_id": 4943,
        "honest_verdict": "blocked_insufficient_v455_mtime_window",
        "mtime_fallback_window": {
            "compute_bound_count": 3,
            "n_arms": 8,
            "wall_minutes": 112.46,
        },
        "preconditions_checked": {
            "window_gate": {
                "min_arms": 10,
                "n_arms": 8,
                "passed": False,
            }
        },
        "research_conductor_modified": False,
    }


def _make_root(root: Path, *, include_next: bool, active_milestone: str = "2026.06.456") -> None:
    _write_text(root / "research-roadmap.yaml", f"milestone: {active_milestone}\n")
    if include_next:
        _write_text(root / "research-roadmap-next.yaml", "milestone: 2026.06.456\n")
    _write_text(root / mod.REGISTRY_REL_PATH, "schema_version: 1\nreproducible_total_levels: 69\n")
    _write_json(
        root / mod.RETRO_REL_PATH,
        {
            "experiments_completed": 0,
            "milestone": "2026.06.455",
            "summary": "False-zero detector gap; on-disk verification shows exp4935-exp4945.",
        },
    )
    _write_json(root / mod.CAPSTONE_REL_PATH, _capstone_artifact())
    _write_json(root / mod.A3_SELF_PLAY_REL_PATH, _self_play_artifact())
    _write_json(root / mod.B3_STAMPING_REL_PATH, _stamping_artifact())


def _runner(calls: list[list[str]], *, roadmap_ok: bool, offline_ok: bool, pretest_ok: bool):
    def run(command: list[str], _root: Path) -> mod.CommandResult:
        calls.append(command)
        command_text = " ".join(command)
        if "research-roadmap-next.yaml" in command_text:
            return mod.CommandResult(command, 0 if roadmap_ok else 1, "ok\n", "missing")
        if "offline_arcade" in command_text:
            return mod.CommandResult(command, 0 if offline_ok else 1, "", "arcade")
        return mod.CommandResult(command, 0 if pretest_ok else 1, "passed", "failed")

    return run


def test_req_capstone_4946_spec_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4946: OpenSpec declares the .455/.456 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-4946") :]

    assert mod._do_not_queue(["representation_5"]) == ["representation_5"]
    assert mod._first_corrigendum({"corrigendum_pending": []}, "DURATION_TOO_SHORT") == {}
    for ref in mod.SPEC_REFS:
        assert ref in section
    assert str(mod.OUTPUT_REL_PATH) in section
    for field in mod.REQUIRED_FIELDS:
        assert field in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_capstone_4946_blocked_missing_roadmap_next_writes_deliverable(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4946-BLOCKED-PRECONDITION: blocked still records .455 facts."""

    _make_root(tmp_path, include_next=False)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=False, offline_ok=True, pretest_ok=True),
        started_s=10.0,
        now_s=10.25,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_next_yaml_missing"
    assert len(calls) == 2
    assert artifact["pretest_gate"] == {
        "green": False,
        "ran": False,
        "reason": "skipped_after_precondition_failure",
    }
    assert artifact["transition"]["active_milestone_confirmed"] == "2026.06.456"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["transition_performed"] is False
    assert artifact["leaderboard_submission"] is False
    assert artifact["arc_first_win_wall_closed_hidden_state"] is True
    assert artifact["deliverable_locked_agent_plus_fover_paper"] is True
    assert artifact["v456_is_final_stretch_plus_pivot_turnkey"] is True
    assert artifact["reproducible_total_levels"] == 69
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["passed"] is False
    assert artifact["preconditions_checked"]["offline_arcade"]["passed"] is True
    assert artifact["close_state_455"]["capstone"]["honest_verdict"] == (
        "complete_capstone_v455_submission_ready_levels_69_heldout_0.04_"
        "package_ready_pivot_executable_7_1"
    )
    assert artifact["close_state_455"]["a1_a2_no_banked"]["no_banked"] is True
    assert artifact["close_state_455"]["a1_a2_no_banked"]["candidate_banks"] == [
        {"game": "lf52", "new_levels_banked": 0, "source": "A1_LEVELUP"},
        {"game": "sb26", "new_levels_banked": 0, "source": "A2_LEVELUP"},
    ]
    assert artifact["close_state_455"]["a1_a2_no_banked"][
        "second_consecutive_flat_milestone"
    ] is True
    assert artifact["close_state_455"]["a4_heldout"]["heldout_first_win_rate"] == 0.04
    assert artifact["close_state_455"]["a4_heldout"]["flag_resolved"] is True
    assert artifact["close_state_455"]["a4_heldout"]["tautology_warn_only"] is True
    assert artifact["close_state_455"]["d_pivot"]["pivot_executable_on_7_1"] is True
    assert artifact["close_state_455"]["d_pivot"]["arxiv_id"] == "2605.18871"
    assert artifact["close_state_455"]["b2_package"]["peak_vram_gb"] == 15.146
    assert artifact["close_state_455"]["b2_package"]["operator_only"] is True
    assert artifact["close_state_455"]["recurring_infra_bugs_to_fix"][0]["kind"] == (
        "DURATION_TOO_SHORT"
    )
    assert artifact["close_state_455"]["recurring_infra_bugs_to_fix"][1]["honest_verdict"] == (
        "blocked_insufficient_v455_mtime_window"
    )
    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4946_complete_transition_runs_pretest(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4946: complete path records .456 active and green pre-test."""

    _make_root(tmp_path, include_next=True)
    calls: list[list[str]] = []
    exit_code = mod.main(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
    )
    artifact = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert exit_code == 0
    assert len(calls) == 3
    assert artifact["honest_verdict"] == (
        "complete_455_archived_456_activated_final_stretch_pivot_turnkey_recorded"
    )
    assert artifact["pretest_gate"]["ran"] is True
    assert artifact["pretest_gate"]["green"] is True
    assert artifact["transition"]["active_milestone_confirmed"] == "2026.06.456"
    assert artifact["transition"]["activation_state"] == "already_active_or_activated_456"
    assert artifact["transition_performed"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4946_blocked_offline_arcade_skips_pretest(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4946-BLOCKED-PRECONDITION: offline arcade failure blocks."""

    _make_root(tmp_path, include_next=True)
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
