"""Tests for REQ-CAPSTONE-4935 / SCENARIO-CAPSTONE-4935."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4935_archive_454_activate_455 as mod


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
            ".454 submission readiness: 69 reproducible levels, no trusted A1/A2 "
            "banks counted; held-out first-win 0.04 with flag_resolved=True; D "
            "reports no trusted efficiency lift; package ready."
        ),
        "heldout_first_win_rate": {
            "flag_resolved": True,
            "games_evaluated": 25,
            "games_remaining": 0,
            "honest_verdict": "complete_heldout_first_win_0.04_full25_live_flag_resolved",
            "live_agent_ran": True,
            "rate": 0.04,
        },
        "honest_verdict": (
            "complete_capstone_v454_submission_maximized_levels_69_heldout_0.04_"
            "package_ready_efficiency_null"
        ),
        "milestone_scorecard": {
            "action_efficiency": {
                "d_honest_verdict": "complete_matm_similarity_retrieval_no_efficiency_gain_retired",
                "decision": "honest_null_not_trusted_lift",
                "retire_if_same_verdict": True,
                "reported_lift": None,
            },
            "banks": {
                "b1_banks_trustworthy": False,
                "candidate_banks": [
                    {"game": "sp80", "new_levels_banked": 0, "source": "A1_LEVELUP"},
                    {"game": "su15", "new_levels_banked": 0, "source": "A2_LEVELUP"},
                ],
                "computed_total": 69,
                "counted": [],
                "registry_total": 69,
            },
            "heldout_go_no_go": {
                "flag_resolved": True,
                "games_evaluated": 25,
                "games_remaining": 0,
                "honest_verdict": "complete_heldout_first_win_0.04_full25_live_flag_resolved",
                "rate": 0.04,
            },
            "submission_package": {
                "decision": "package_ready_operator_only",
                "honest_verdict": "success_submission_package_ready_final_pre_deadline",
                "operator_only": True,
                "peak_vram_gb": 15.146,
                "ready": True,
                "submits": False,
            },
            "wall_closure": {
                "closed": True,
                "closure_verdict": "WALL_IS_HIDDEN_STATE",
            },
        },
        "post_sprint_pivot": {
            "decision": "post_6_30_distributional_energy_verifier_pivot",
            "deliverable": "current ~0.05 agent + publishable FoVer paper",
            "do_not_queue": "representation_5",
        },
        "reproducible_total_levels": 69,
        "submission_package_ready": {
            "decision": "package_ready_operator_only",
            "honest_verdict": "success_submission_package_ready_final_pre_deadline",
            "operator_only": True,
            "peak_vram_gb": 15.146,
            "ready": True,
            "submits": False,
        },
    }


def _make_root(root: Path, *, include_next: bool, active_milestone: str = "2026.06.455") -> None:
    _write_text(root / "research-roadmap.yaml", f"milestone: {active_milestone}\n")
    if include_next:
        _write_text(root / "research-roadmap-next.yaml", "milestone: 2026.06.455\n")
    _write_text(root / mod.REGISTRY_REL_PATH, "schema_version: 1\nreproducible_total_levels: 69\n")
    _write_json(
        root / mod.RETRO_REL_PATH,
        {
            "experiments_completed": 0,
            "milestone": "2026.06.454",
            "summary": "Detector false-zeroed, but exp4924-exp4934 exist on disk.",
        },
    )
    _write_json(root / mod.CAPSTONE_REL_PATH, _capstone_artifact())


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


def test_req_capstone_4935_spec_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4935: OpenSpec declares the .454/.455 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-4935") :]

    assert mod._do_not_queue(["representation_5"]) == ["representation_5"]
    for ref in mod.SPEC_REFS:
        assert ref in section
    assert str(mod.OUTPUT_REL_PATH) in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_capstone_4935_blocked_missing_roadmap_next_writes_deliverable(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4935-BLOCKED-PRECONDITION: blocked still records .454 facts."""

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
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["transition_performed"] is False
    assert artifact["leaderboard_submission"] is False
    assert artifact["arc_first_win_wall_closed_hidden_state"] is True
    assert artifact["deliverable_locked_agent_plus_fover_paper"] is True
    assert artifact["v455_is_final_sprint_plus_pivot_readiness"] is True
    assert artifact["reproducible_total_levels"] == 69
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["passed"] is False
    assert artifact["preconditions_checked"]["offline_arcade"]["passed"] is True
    assert artifact["close_state_454"]["capstone"]["honest_verdict"].startswith(
        "complete_capstone_v454_submission_maximized_levels_69"
    )
    assert artifact["close_state_454"]["a1_a2_no_banked"]["no_banked"] is True
    assert artifact["close_state_454"]["a1_a2_no_banked"]["candidate_banks"] == [
        {"game": "sp80", "new_levels_banked": 0, "source": "A1_LEVELUP"},
        {"game": "su15", "new_levels_banked": 0, "source": "A2_LEVELUP"},
    ]
    assert artifact["close_state_454"]["a4_heldout"]["heldout_first_win_rate"] == 0.04
    assert artifact["close_state_454"]["a4_heldout"]["flag_resolved"] is True
    assert artifact["close_state_454"]["d_efficiency"]["retired"] is True
    assert artifact["close_state_454"]["b2_package"]["peak_vram_gb"] == 15.146
    assert "representation_5" in artifact["close_state_454"]["do_not_queue"]
    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4935_complete_transition_runs_pretest(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4935: complete path records .455 active and green pre-test."""

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
        "complete_454_archived_455_activated_final_sprint_pivot_readiness_recorded"
    )
    assert artifact["pretest_gate"]["ran"] is True
    assert artifact["pretest_gate"]["green"] is True
    assert artifact["transition"]["active_milestone_confirmed"] == "2026.06.455"
    assert artifact["transition"]["activation_state"] == "already_active_or_activated_455"
    assert artifact["transition_performed"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4935_blocked_offline_arcade_skips_pretest(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4935-BLOCKED-PRECONDITION: offline arcade failure blocks."""

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
