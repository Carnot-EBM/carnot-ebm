"""Tests for REQ-REPORT-4881 / SCENARIO-REPORT-4881."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4881_archive_449_activate_450 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _capstone() -> JsonDict:
    return {
        "honest_verdict": "complete_a1_generation_wall_non_test_capstone_ready",
        "a1_generation_wall_fork_verdict": {
            "upstream_honest_verdict": (
                "complete_generation_wall_fork_probe_retired_positive_control_failed"
            ),
            "verdict": "non_test_b1_untrusted",
            "fork_verdict": None,
            "computed_fork_verdict": "INDUCER_CEILING",
            "b1_trusted": False,
            "a1_genuinely_diagnostic": False,
            "next_450_pivot": "do_not_use_a1_non_test",
            "positive_control_game": "tu93",
            "positive_control_migrated": False,
            "trust_checks": {
                "a1_ran_live_on_gpu0": True,
                "planner_blind": True,
                "positive_control_migrated": False,
                "numbers_match_fork": False,
            },
            "a1_failure_reasons": [
                "positive_control_not_migrated",
                "positive_control_not_covered",
                "positive_control_low_accuracy",
                "fork_verdict_missing",
            ],
            "checks": {
                "a1_live_gpu": {
                    "duration_s": 587.0132,
                    "generator_backend": "gpu0_cuda",
                    "passed": True,
                },
                "a1_live_path": {"passed": True},
                "a1_planner_blind_to_banked_answer": {"passed": True},
                "a1_positive_control": {
                    "positive_control_game": "tu93",
                    "engine_heldout_accuracy": 0.0,
                    "passed": False,
                },
            },
        },
        "a1b_inducer_swing": {
            "cegis_heldout_accuracy_delta_median": 0.0,
            "delta_ci95": [0.0, 0.0],
            "positive_control_passed": False,
            "ran": True,
            "status": "ran",
        },
        "levelup_bank": {
            "target_game": "s5i5",
            "new_levels_banked": 1,
            "reproducible_total_levels_after": 67,
            "solve_provenance": "live_agent_self_discovery",
        },
        "self_play_checkpoint": {
            "target_game": "re86",
            "verifier_checkpoint_refreshed": True,
            "checkpoint_path": "models/arc_verifier_re86.json",
            "offline_reproduced": True,
        },
        "heldout_readiness": {
            "heldout_first_win_rate": 0.04,
            "heldout_first_win_ci": {"ci95": [0.0, 0.0]},
            "positive_control_passed": True,
            "live_agent_ran": True,
            "generator_backend": "gpu0_cuda",
        },
        "submission_package_state": {
            "submission_package_ready": True,
            "operator_only": True,
            "vram_estimate_gb": 15.146,
        },
        "hardware_continuity": {
            "kv260_ssh_reachable": True,
            "board_state": {"uio_device_count": 5},
        },
        "sota_handoff": {
            "aimed_at_fork_verdict": "INDUCER_CEILING",
            "sota_to_experiment_mapping_note": {
                "source_ids": [
                    "2506.02918",
                    "2509.03956",
                    "2507.15877",
                    "2605.05138",
                    "2507.03160",
                    "2203.13474",
                    "2606.25421",
                    "2606.26217",
                ]
            },
        },
    }


def _corrigendum() -> JsonDict:
    return {
        "honest_verdict": (
            "complete_fork_probe_0667_unreproduced_honest_changing_acc_0.0"
        ),
        "mean_changing_acc_prior": 0.0,
        "fork_probe_number_is_artifact": True,
        "per_game": [
            {
                "game": "ka59",
                "cell_recall_prior_engine": 0.8853,
                "changing_acc_prior_engine": 0.0,
            },
            {
                "game": "tn36",
                "cell_recall_prior_engine": 0.8633,
                "changing_acc_prior_engine": 0.0,
            },
            {
                "game": "sc25",
                "cell_recall_prior_engine": 0.7506,
                "changing_acc_prior_engine": 0.0,
            },
            {
                "game": "cd82",
                "cell_recall_prior_engine": 0.2839,
                "changing_acc_prior_engine": 0.0,
            },
            {
                "game": "lp85",
                "cell_recall_prior_engine": 0.2574,
                "changing_acc_prior_engine": 0.0,
            },
        ],
    }


def _make_root(root: Path, *, include_next: bool) -> None:
    _write_text(
        root / "research-roadmap.yaml",
        "milestone: 2026.06.450\nnote: Energy CONCLUDED; do NOT re-propose energy stages.\n",
    )
    if include_next:
        _write_text(
            root / "research-roadmap-next.yaml",
            "milestone: 2026.06.450\nnote: Energy CONCLUDED.\n",
        )
    _write_text(root / mod.REGISTRY_REL_PATH, "schema_version: 1\nreproducible_total_levels: 67\n")
    _write_json(root / mod.CAPSTONE_REL_PATH, _capstone())
    _write_json(root / mod.CORRIGENDUM_REL_PATH, _corrigendum())


def _runner(calls: list[list[str]], *, roadmap_ok: bool, offline_ok: bool, pretest_ok: bool):
    def run(command: list[str], _root: Path) -> mod.CommandResult:
        calls.append(command)
        code = " ".join(command)
        if "research-roadmap-next.yaml" in code:
            return mod.CommandResult(command, 0 if roadmap_ok else 1, "ok\n", "" if roadmap_ok else "missing")
        if "offline_arcade" in code:
            return mod.CommandResult(command, 0 if offline_ok else 1, "", "" if offline_ok else "arcade")
        return mod.CommandResult(command, 0 if pretest_ok else 1, "passed" if pretest_ok else "failed", "")

    return run


def test_req_report_4881_spec_declares_transition_contract() -> None:
    """REQ-REPORT-4881: OpenSpec declares the .449/.450 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert str(mod.OUTPUT_REL_PATH) in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_report_4881_blocked_missing_roadmap_next_writes_deliverable(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4881: blocked roadmap-next still records required facts."""

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
    assert all(command[:2] == [".venv/bin/python", "-c"] for command in calls)
    assert artifact["pretest_gate"] == {
        "ran": False,
        "green": False,
        "reason": "skipped_after_precondition_failure",
    }
    assert artifact["transition_performed"] is False
    assert artifact["a449_fork_untrusted_non_test"] is True
    assert artifact["corrigendum_change_location_learnable"] is True
    assert artifact["exact_match_metric_was_degenerate"] is True
    assert artifact["energy_program_concluded"] is True
    assert artifact["reproducible_total_levels"] == 67
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["passed"] is False
    assert artifact["preconditions_checked"]["offline_arcade"]["passed"] is True
    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_4881_complete_transition_records_corrected_close_state(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4881: complete path records the corrected .449 state."""

    _make_root(tmp_path, include_next=True)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
        started_s=20.0,
        now_s=20.125,
    )

    assert artifact["honest_verdict"] == (
        "complete_449_archived_450_activated_a1_untrusted_non_test_value_gap_focus"
    )
    assert calls[-1] == [".venv/bin/pytest", "tests/python", "-q"]
    assert artifact["transition_performed"] is True
    assert artifact["pretest_gate"]["green"] is True
    assert artifact["active_milestone_confirmed"] == "2026.06.450"
    assert artifact["a449_close_state"]["a1"]["generator_backend"] == "gpu0_cuda"
    assert artifact["a449_close_state"]["a1"]["positive_control_exact_accuracy"] == 0.0
    assert artifact["a449_close_state"]["corrigendum"]["max_cell_recall"] == 0.8853
    assert artifact["a449_close_state"]["a1b"]["cegis_heldout_accuracy_delta_median"] == 0.0
    assert artifact["a449_close_state"]["a4"]["heldout_first_win_rate"] == 0.04
    assert artifact["a449_close_state"]["b2"]["vram_estimate_gb"] == 15.146
    assert artifact["a449_close_state"]["c"]["uio_device_count"] == 5
    assert len(artifact["cited_upstream_artifacts"]) == 3
    assert mod.validate_artifact(artifact) == []


def test_validate_artifact_and_helpers_cover_failure_cases(tmp_path: Path) -> None:
    """REQ-REPORT-4881: malformed transition artifacts fail validation."""

    _make_root(tmp_path, include_next=True)
    artifact = mod.build_artifact(
        root=tmp_path,
        honest_verdict="complete_449_archived_450_activated_a1_untrusted_non_test_value_gap_focus",
        preconditions_checked={
            "research_roadmap_next_yaml": {"passed": True},
            "offline_arcade": {"passed": True},
        },
        pretest_gate={"ran": True, "green": True},
        transition_performed=True,
        duration_s=0.0001,
    )

    assert mod.validate_artifact(artifact) == []
    assert "missing_field:honest_verdict" in mod.validate_artifact(
        {key: value for key, value in artifact.items() if key != "honest_verdict"}
    )
    assert "honest_verdict_missing_terminal_prefix" in mod.validate_artifact(
        {**artifact, "honest_verdict": "done"}
    )
    assert "invalid_inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_llm"}
    )
    assert "missing_principle:energy_program_concluded" in mod.validate_artifact(
        {**artifact, "field_principles": {}}
    )
    assert "invalid_reproducible_total_levels" in mod.validate_artifact(
        {**artifact, "reproducible_total_levels": "67"}
    )
    assert "invalid_a449_fork_untrusted_non_test" in mod.validate_artifact(
        {**artifact, "a449_fork_untrusted_non_test": False}
    )
    assert "invalid_reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )
    assert mod.precondition_blocker(tmp_path, {"research_roadmap_next_yaml": {"passed": False}}) == (
        "blocked_research_roadmap_next_yaml_poison"
    )
    assert mod.precondition_blocker(
        tmp_path / "missing", {"research_roadmap_next_yaml": {"passed": False}}
    ) == "blocked_research_roadmap_next_yaml_missing"
    assert mod.precondition_blocker(
        tmp_path, {"research_roadmap_next_yaml": {"passed": True}, "offline_arcade": {"passed": False}}
    ) == "blocked_offline_arcade_unavailable"
    assert mod.precondition_blocker(
        tmp_path, {"research_roadmap_next_yaml": {"passed": True}, "offline_arcade": {"passed": True}}
    ) == ""
    assert mod.command_summary(mod.CommandResult(["x"], 2, "out", "err")) == {
        "command": ["x"],
        "exit_code": 2,
        "passed": False,
        "stdout_tail": "out",
        "stderr_tail": "err",
    }
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_yaml_object(tmp_path / "missing.yaml") == {}
    assert mod.file_sha256(tmp_path / "missing.txt") == ""
    no_milestone_root = tmp_path / "no_milestone"
    _write_text(no_milestone_root / "research-roadmap.yaml", "note: no active milestone\n")
    assert mod.read_active_milestone(no_milestone_root) == ("unknown", "research-roadmap.yaml")
    assert mod.read_active_milestone(tmp_path / "missing") == ("unknown", "research-roadmap.yaml")

    inactive_root = tmp_path / "inactive"
    _make_root(inactive_root, include_next=True)
    _write_text(
        inactive_root / "research-roadmap.yaml",
        "milestone: 2026.06.449\nnote: Energy CONCLUDED.\n",
    )
    inactive_calls: list[list[str]] = []
    inactive = mod.run(
        root=inactive_root,
        command_runner=_runner(inactive_calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
        started_s=1.0,
        now_s=1.1,
    )
    assert inactive["honest_verdict"] == "blocked_450_not_active"
    assert inactive["pretest_gate"] == {
        "ran": False,
        "green": False,
        "reason": "skipped_until_450_active",
    }
    assert len(inactive_calls) == 2

    pretest_fail_root = tmp_path / "pretest_fail"
    _make_root(pretest_fail_root, include_next=True)
    pretest_calls: list[list[str]] = []
    pretest_failed = mod.run(
        root=pretest_fail_root,
        command_runner=_runner(pretest_calls, roadmap_ok=True, offline_ok=True, pretest_ok=False),
        started_s=2.0,
        now_s=2.1,
    )
    assert pretest_failed["honest_verdict"] == "blocked_pretest_gate_failed"
    assert pretest_failed["pretest_gate"]["ran"] is True
    assert pretest_failed["pretest_gate"]["green"] is False
    assert pretest_calls[-1] == mod.PRETEST_COMMAND


def test_main_and_default_command_runner(tmp_path: Path) -> None:
    """REQ-REPORT-4881: script entrypoint writes the artifact path."""

    _make_root(tmp_path, include_next=True)
    calls: list[list[str]] = []
    exit_code = mod.main(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
    )
    default_result = mod.run_command([".venv/bin/python", "-c", "print('ok')"], REPO)

    assert exit_code == 0
    assert (tmp_path / mod.OUTPUT_REL_PATH).exists()
    assert default_result.exit_code == 0
    assert default_result.stdout == "ok\n"
