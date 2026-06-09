"""Tests for Exp 3974 .367 archive and .368 activation.

Spec refs: REQ-REPORT-3974, SCENARIO-REPORT-3974,
SCENARIO-REPORT-3974-BLOCKED-YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v367_activate_v368_3974 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


SUMMARY_STDOUT = """\
==============================================================================
ARTIFACT  experiment_3963_archive_v366_activate_v367.json
------------------------------------------------------------------------------
  verdict          : complete: archived_v366_v367_active_second_solve_m3_fabrication_recorded_arc_substrate_green
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 15.060667   substrate: aggregation_from_upstream_artifacts
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3964_r11l_incremental_l2.json
------------------------------------------------------------------------------
  verdict          : complete: r11l_levels1_of6_first_fail2
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.77   substrate: offline_arc_agi3_perception_planner_real_env_confirmed
  headline metrics :
      ACCURACY_levels_solved = 1
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3965_lp85_incremental_l2.json
------------------------------------------------------------------------------
  verdict          : complete: lp85_levels1_first_fail2
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 1.344   substrate: offline_arc_agi3_perception_planner_real_env_confirmed
  headline metrics :
      ACCURACY_levels_solved = 1
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3966_third_game_first_solve.json
------------------------------------------------------------------------------
  verdict          : complete: third_game_solve_sc25-635fd71a_levels1_solvedTrue
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 374.3   substrate: offline_arc_agi3_perception_planner_real_env_confirmed
  headline metrics :
      ACCURACY_levels_solved = 1
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3967_m3_honest_efficiency.json
------------------------------------------------------------------------------
  verdict          : blocked_verifier_not_in_loop
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.0   substrate: offline_air_gapped_arc_agi3_local_environments
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3968_active_codex_nonspatial_sweep.json
------------------------------------------------------------------------------
  verdict          : complete: exp3968_active_codex_nonspatial_sweep_trustworthy_0of6
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 1082.4   substrate: offline_arc_agi3_plus_codex_program_synthesis_consistency_verified
  headline metrics :
      vc33_baseline_energy = 0.005
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3969_hidden_state_pinductor.json
------------------------------------------------------------------------------
  verdict          : complete: pinductor_latents_no_drop_energy
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 28.943685293197632   substrate: offline_arc_agi3_pinductor
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3970_cross_game_arcmemo_transfer.json
------------------------------------------------------------------------------
  verdict          : success: arcmemo_transfer_win_reused_2_later_games
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.4   substrate: offline_arc_agi3_existing_codex_sweep_plus_arcmemo_concept_memory
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3971_m4_offline_quota_gate.json
------------------------------------------------------------------------------
  verdict          : success: quota_gate_cleared_hybrid_levels3_baseline0_prior0_operator_ready
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 16.843   substrate: offline_arc_agi3_hybrid_policy_quota_gate_local_env
  headline metrics :
      hybrid_accuracy_levels_solved = 3
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3972_hardware_continuity.json
------------------------------------------------------------------------------
  verdict          : complete: hardware_continuity_3972_kvreachable_overlay_absent_gmblocked_gatemate_unreachable_pfreachable_ssh_continuity_recorded
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 7.794915   substrate: hardware_smoke
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3973_capstone_v367.json
------------------------------------------------------------------------------
  verdict          : complete: capstone_v367_accuracy_progress1_total_real_levels3_verifier_earns_efficiencyfalse_missing0_flagged_skipped0
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.32506837596884   substrate: aggregation_from_upstream_artifacts_via_summarize_artifact_py
  headline metrics :
      accuracy_progress_vs_v366_baseline = 1
  adversarial flags: none
==============================================================================
"""


def _command_result(
    command: list[str],
    *,
    exit_code: int = 0,
    stdout: str = "ok\n",
    stderr: str = "",
) -> mod.CommandResult:
    return mod.CommandResult(command=command, exit_code=exit_code, stdout=stdout, stderr=stderr)


def _summary_result(exit_code: int = 0, stdout: str = SUMMARY_STDOUT) -> mod.CommandResult:
    return _command_result(mod.summary_command(), exit_code=exit_code, stdout=stdout)


def _import_stdout(*, all_ok: bool = True) -> str:
    return json.dumps(
        {
            module: {"import_ok": all_ok, "error": None if all_ok else "ImportError"}
            for module in mod.ARC_IMPORT_MODULES
        },
        sort_keys=True,
    )


def _import_result(*, all_ok: bool = True) -> mod.CommandResult:
    return _command_result(
        mod.arc_modules_import_command(),
        exit_code=0 if all_ok else 1,
        stdout=_import_stdout(all_ok=all_ok),
    )


def _seed_repo(
    root: Path,
    *,
    corrupt_complete: bool = False,
    milestone: str = "2026.06.368",
    manifest: str | None = None,
    gap4: str | None = None,
) -> None:
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{milestone}"\n'
        "tasks:\n"
        "  - id: exp3974-archive-v367-activate-v368\n",
        encoding="utf-8",
    )
    complete_text = (
        "milestones:\n"
        "- id: 2026.06.367\n"
        "  title: stale conductor archive\n"
        "  completed: '2026-06-09'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3966-third-game-first-solve\n"
        "    deliverable: results/experiment_3966_third_game_first_solve.json\n"
        "    result: OK (conductor)\n"
        "  - id: exp3967-m3-honest-efficiency\n"
        "    deliverable: results/experiment_3967_m3_honest_efficiency.json\n"
        "    result: OK (conductor)\n"
    )
    if corrupt_complete:
        complete_text += "  - id: poison\n    result: complete: unquoted colon\n"
    (root / "research-complete.yaml").write_text(complete_text, encoding="utf-8")
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "experiments").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        manifest
        or (
            "retired_experiments:\n"
            "  - retirement_marker: gap3_trained_content_energy_selector_retired_stage2v2_2026_06_09\n"
            "    reason: existing\n"
        ),
        encoding="utf-8",
    )
    (root / "ops" / "verifier_gaps.md").write_text(
        gap4
        or (
            "### GAP-4: same-shape rule-application consistency\n"
            "- candidate design: execution/program-synthesis verification\n"
        ),
        encoding="utf-8",
    )
    (root / "ops" / "changelog.md").write_text("changelog before\n", encoding="utf-8")
    (root / "ops" / "status.md").write_text("status before\n", encoding="utf-8")
    (root / "_bmad" / "traceability.md").write_text("trace before\n", encoding="utf-8")
    (root / "scripts" / "research_conductor.py").write_text("# conductor before\n", encoding="utf-8")


def _run_success(root: Path, **overrides: object) -> Path:
    kwargs = {
        "research_complete_parse_result": _command_result(mod.research_complete_yaml_command()),
        "summary_result": _summary_result(),
        "arc_substrate_test_result": _command_result(mod.arc_substrate_test_command()),
        "arc_modules_import_result": _import_result(),
        "started_s": 1.0,
        "now_s": 2.25,
    }
    kwargs.update(overrides)
    return mod.run(root, **kwargs)


def test_req_report_3974_spec_anchor_exists() -> None:
    """REQ-REPORT-3974: OpenSpec declares the .367 archive contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3974" in spec
    assert "SCENARIO-REPORT-3974" in spec
    assert "SCENARIO-REPORT-3974-BLOCKED-YAML" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "prior_three_games_solved_recorded" in spec
    assert "prior_m3_still_open_recorded" in spec
    assert "gap4_spec_present" in spec


def test_scenario_report_3974_run_appends_truth_and_green_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3974: archive .367 and preserve GAP-4 readiness."""

    _seed_repo(tmp_path)
    before = {
        "complete": (tmp_path / "research-complete.yaml").read_text(encoding="utf-8"),
        "manifest": (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8"),
        "roadmap": (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8"),
        "changelog": (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8"),
        "status": (tmp_path / "ops" / "status.md").read_text(encoding="utf-8"),
        "trace": (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8"),
        "conductor": (tmp_path / "scripts" / "research_conductor.py").read_text(encoding="utf-8"),
    }

    out_path = _run_success(tmp_path)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    complete_text = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    archived = yaml.safe_load(complete_text)["milestones"][-1]
    task_results = {task["id"]: task["result"] for task in archived["tasks"]}

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.terminal_verdict()
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert not isinstance(artifact[field], dict), field
    assert artifact["archived_milestone"] == "2026.06.367"
    assert artifact["activated_milestone"] == "2026.06.368"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["arc_substrate_tests_green"] is True
    assert artifact["arc_modules_importable"] is True
    assert artifact["prior_three_games_solved_recorded"] is True
    assert artifact["prior_m3_still_open_recorded"] is True
    assert artifact["gap4_spec_present"] is True
    assert artifact["gap3_lineage_retired_recorded"] is True
    assert artifact["n_tasks_archived"] == 11
    assert artifact["duration_s"] == 1.25
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert "exp3966: complete: third_game_solve_sc25-635fd71a" in artifact[
        "prior_milestone_verdicts_summary"
    ]
    assert "exp3967: blocked_verifier_not_in_loop" in artifact[
        "prior_milestone_verdicts_summary"
    ]
    assert "exp3968: complete: exp3968_active_codex_nonspatial_sweep_trustworthy_0of6" in artifact[
        "prior_milestone_verdicts_summary"
    ]
    assert artifact["arc_module_import_results"]["carnot.agentic.arc_agi3_action_efficiency"][
        "import_ok"
    ] is True

    assert complete_text.startswith(before["complete"].rstrip())
    assert complete_text.count(mod.ARCHIVE_MARKER) == 1
    assert complete_text.count("- id: 2026.06.367") == 2
    assert "result: complete:" not in complete_text
    assert "result: 'complete:" in complete_text
    assert archived["activation_recorded"] == "exp3974-archive-v367-activate-v368"
    assert "R11L_SOLVED" in task_results["exp3964-r11l-incremental-l2"]
    assert "LP85_SOLVED" in task_results["exp3965-lp85-incremental-l2"]
    assert "SC25_SOLVED" in task_results["exp3966-third-game-first-solve"]
    assert "M3_STILL_OPEN" in task_results["exp3967-m3-honest-efficiency"]
    assert "WORLD_MODEL_TRUSTWORTHY_0OF6" in task_results[
        "exp3968-active-codex-nonspatial-sweep"
    ]
    assert yaml.safe_load(complete_text)

    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(
        encoding="utf-8"
    ) == before["manifest"]
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before["roadmap"]
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before["changelog"]
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before["status"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8") == before["trace"]
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before["conductor"]


def test_req_report_3974_append_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3974: rerunning does not duplicate the archive entry."""

    _seed_repo(tmp_path)

    first = _run_success(tmp_path).read_text(encoding="utf-8")
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second = _run_success(tmp_path).read_text(encoding="utf-8")

    assert first == second
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == first_complete
    assert first_complete.count(mod.ARCHIVE_MARKER) == 1


def test_scenario_report_3974_blocked_yaml_writes_artifact_without_edits(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3974-BLOCKED-YAML: corrupt YAML exits before edits."""

    _seed_repo(tmp_path, corrupt_complete=True)
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_manifest = (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8")

    out_path = mod.run(
        tmp_path,
        research_complete_parse_result=_command_result(
            mod.research_complete_yaml_command(),
            exit_code=1,
            stderr="yaml parser failed",
        ),
        summary_result=_summary_result(),
        arc_substrate_test_result=_command_result(mod.arc_substrate_test_command()),
        arc_modules_import_result=_import_result(),
        started_s=7.0,
        now_s=7.1,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison")
    assert artifact["research_complete_yaml_parses"] is False
    assert artifact["exclusion_manifest_parses"] is False
    assert artifact["arc_substrate_tests_green"] is False
    assert artifact["arc_modules_importable"] is False
    assert artifact["prior_three_games_solved_recorded"] is False
    assert artifact["prior_m3_still_open_recorded"] is False
    assert artifact["gap4_spec_present"] is False
    assert artifact["preconditions_checked"]["research_complete_yaml_exists"] is True
    assert artifact["preconditions_checked"]["research_complete_yaml_parsed_before"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8") == before_manifest


def test_req_report_3974_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3974: hard helper failures block instead of fabricating success."""

    _seed_repo(tmp_path, milestone="2026.06.367")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_v368_not_active")

    _seed_repo(tmp_path, manifest="retired: [\n")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_exclusion_manifest_yaml_poison")

    _seed_repo(tmp_path)
    (tmp_path / "ops" / "exclusion_manifest.yaml").unlink()
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_exclusion_manifest_missing")

    _seed_repo(tmp_path, manifest="retired_experiments: []\n")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_gap3_retired_lineage_missing")

    _seed_repo(tmp_path, gap4="### GAP-3: old gap\n")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_gap4_spec_missing")

    _seed_repo(tmp_path)
    artifact = json.loads(
        _run_success(tmp_path, summary_result=_summary_result(exit_code=127, stdout="")).read_text(
            encoding="utf-8"
        )
    )
    assert artifact["honest_verdict"].startswith("blocked_v367_summary_command_failed")

    _seed_repo(tmp_path)
    no_three = SUMMARY_STDOUT.replace("complete: r11l_levels1_of6_first_fail2", "blocked_no_r11l")
    artifact = json.loads(
        _run_success(tmp_path, summary_result=_summary_result(stdout=no_three)).read_text(
            encoding="utf-8"
        )
    )
    assert artifact["honest_verdict"].startswith("blocked_prior_three_games_solved_missing")

    _seed_repo(tmp_path)
    no_m3 = SUMMARY_STDOUT.replace("blocked_verifier_not_in_loop", "complete: pruner_helps")
    artifact = json.loads(
        _run_success(tmp_path, summary_result=_summary_result(stdout=no_m3)).read_text(
            encoding="utf-8"
        )
    )
    assert artifact["honest_verdict"].startswith("blocked_prior_m3_still_open_missing")

    _seed_repo(tmp_path)
    artifact = json.loads(
        _run_success(
            tmp_path,
            arc_substrate_test_result=_command_result(mod.arc_substrate_test_command(), exit_code=1),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_arc_substrate_tests_failed")
    assert artifact["arc_substrate_tests_green"] is False

    _seed_repo(tmp_path)
    artifact = json.loads(
        _run_success(tmp_path, arc_modules_import_result=_import_result(all_ok=False)).read_text(
            encoding="utf-8"
        )
    )
    assert artifact["honest_verdict"].startswith("blocked_arc_module_import")
    assert artifact["arc_modules_importable"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("prior_three_games_solved_recorded"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p.update(archived_milestone="2026.06.366"), "archived milestone"),
        (lambda p: p.update(activated_milestone="2026.06.367"), "activated milestone"),
        (lambda p: p.update(research_complete_yaml_parses=False), "research-complete"),
        (lambda p: p.update(exclusion_manifest_parses=False), "manifest"),
        (lambda p: p.update(arc_substrate_tests_green=False), "ARC substrate"),
        (lambda p: p.update(arc_modules_importable=False), "ARC module imports"),
        (lambda p: p.update(prior_three_games_solved_recorded=False), "three games"),
        (lambda p: p.update(prior_m3_still_open_recorded=False), "M3 still open"),
        (lambda p: p.update(gap4_spec_present=False), "GAP-4"),
        (lambda p: p.update(gap3_lineage_retired_recorded=False), "GAP-3"),
        (lambda p: p.update(active_milestone_confirmed=False), "active milestone"),
        (lambda p: p.update(n_tasks_archived=10), "n_tasks_archived"),
        (lambda p: p.update(prior_milestone_verdicts_summary="exp3963: ok"), "missing exp3964"),
        (lambda p: p.update(inference_substrate="live_model"), "inference"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(copied_marker="CUDA"), "compute-bound markers"),
    ],
)
def test_req_report_3974_validate_artifact_rejects_regressions(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3974: validation rejects fields that hide transition risk."""

    _seed_repo(tmp_path)
    payload = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))

    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(broken)


def test_req_report_3974_summary_and_import_helpers() -> None:
    """REQ-REPORT-3974: summaries preserve solve truth, open M3, and imports."""

    records = mod.parse_summary_records(SUMMARY_STDOUT)
    verdicts = mod.task_verdicts_from_summary(SUMMARY_STDOUT)
    summary = mod.build_prior_verdicts_summary(verdicts)
    critical_summary = SUMMARY_STDOUT.replace("LIVE re-check: clean", "LIVE re-check: CRITICAL", 1)
    critical_verdicts = mod.task_verdicts_from_summary(critical_summary)

    assert records["3966"]["duration_s"] == pytest.approx(374.3)
    assert "R11L_SOLVED" in verdicts["exp3964-r11l-incremental-l2"]
    assert "LP85_SOLVED" in verdicts["exp3965-lp85-incremental-l2"]
    assert "SC25_SOLVED" in verdicts["exp3966-third-game-first-solve"]
    assert "M3_STILL_OPEN" in verdicts["exp3967-m3-honest-efficiency"]
    assert "WORLD_MODEL_TRUSTWORTHY_0OF6" in verdicts[
        "exp3968-active-codex-nonspatial-sweep"
    ]
    assert "LIVE_CRITICAL" in critical_verdicts["exp3963-archive-v366-activate-v367"]
    assert "exp3973: complete: capstone_v367" in summary
    assert mod.yaml_single_quote("complete: ok") == "'complete: ok'"
    assert mod.duration_from(None, None) == 0.0001
    assert mod._milestone_from_text("tasks: []\n") == "unknown"
    assert mod.three_games_solved_recorded_from_text("milestones: [\n") is False
    assert mod.three_games_solved_recorded_from_text("milestones:\n- id: 2026.06.366\n") is False
    assert mod.m3_still_open_recorded_from_text("tasks: []\n") is False
    assert mod.m3_still_open_recorded_from_text("milestones:\n- id: 2026.06.367\n  tasks: nope\n") is False
    assert mod.gap4_spec_present_from_text("### GAP-4: x\nexecution/program-synthesis") is True
    assert mod.gap4_spec_present_from_text("### GAP-4: x\n") is False
    assert mod.gap3_lineage_retired_from_text("retirement_marker: x\n") is False
    assert (
        mod.gap3_lineage_retired_from_text(
            "retirement_marker: gap3_trained_content_energy_selector_retired_stage2v2_2026_06_09\n"
        )
        is True
    )
    parsed = mod.parse_arc_module_imports(_import_result())
    assert parsed["carnot.agentic.arc_agi3_action_efficiency"]["import_ok"] is True
    malformed = _command_result(mod.arc_modules_import_command(), stdout="{not json", exit_code=1)
    assert "unparseable" in str(
        mod.parse_arc_module_imports(malformed)["carnot.agentic.arc_world_model_dsl"]["error"]
    )
    partial = _command_result(
        mod.arc_modules_import_command(),
        stdout=json.dumps({"carnot.agentic.arc_agi3_world_model": {"import_ok": True}}),
    )
    assert mod.parse_arc_module_imports(partial)["carnot.agentic.arc_agi3_action_efficiency"][
        "import_ok"
    ] is False


def test_req_report_3974_edge_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3974: defensive fallback paths stay explicit and covered."""

    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    odd_summary = """\
==============================================================================
ARTIFACT  experiment_3967_m3_honest_efficiency.json
------------------------------------------------------------------------------
  verdict          : blocked_verifier_not_in_loop
  flagged_adversarial (stamped): True   |   LIVE re-check: warn
  duration_s       : not-a-number   substrate: aggregation
  adversarial flags: none
==============================================================================
"""
    records = mod.parse_summary_records(odd_summary)
    verdicts = mod.task_verdicts_from_summary(odd_summary)
    assert records["3967"]["duration_s"] is None
    assert "M3_STILL_OPEN" in verdicts["exp3967-m3-honest-efficiency"]
    assert "summarize_artifact stamped_flagged" in verdicts["exp3967-m3-honest-efficiency"]
    assert "missing_artifact:" in verdicts["exp3966-third-game-first-solve"]

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").unlink()
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison_missing")

    _seed_repo(tmp_path)
    monkeypatch.setattr(mod, "append_research_complete_record", lambda *args: "milestones: [\n")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_append_invalid")


def test_req_report_3974_subprocess_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3974: subprocess helpers use the mandated commands."""

    calls: list[list[str]] = []

    class Completed:
        returncode = 0
        stdout = _import_stdout()
        stderr = ""

    def run_subprocess(cmd: list[str], **kwargs: object) -> Completed:
        calls.append(cmd)
        assert kwargs["cwd"] == tmp_path
        return Completed()

    monkeypatch.setattr(mod.subprocess, "run", run_subprocess)
    assert mod.run_research_complete_parse_check(tmp_path).command == (
        mod.research_complete_yaml_command()
    )
    assert calls == [mod.research_complete_yaml_command()]

    calls.clear()
    assert mod.run_summarize_artifacts(tmp_path).stdout == _import_stdout()
    assert calls == [mod.summary_command()]

    calls.clear()
    assert mod.run_arc_substrate_tests(tmp_path).exit_code == 0
    assert calls == [mod.arc_substrate_test_command()]

    calls.clear()
    assert mod.run_arc_modules_import_check(tmp_path).stdout == _import_stdout()
    assert calls == [mod.arc_modules_import_command()]

    def fail_subprocess(cmd: list[str], **kwargs: object) -> object:
        raise mod.subprocess.CalledProcessError(1, cmd, output="", stderr="failed")

    monkeypatch.setattr(mod.subprocess, "run", fail_subprocess)
    assert mod.run_arc_substrate_tests(tmp_path).exit_code == 1
    assert mod.run_arc_modules_import_check(tmp_path).exit_code == 1

    def os_error_subprocess(cmd: list[str], **kwargs: object) -> object:
        raise OSError("missing executable")

    monkeypatch.setattr(mod.subprocess, "run", os_error_subprocess)
    assert mod.run_research_complete_parse_check(tmp_path).exit_code == 127
    assert mod.run_summarize_artifacts(tmp_path).exit_code == 127


def test_scenario_report_3974_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3974: the requested script entrypoint exists."""

    script = Path("scripts/experiments/experiment_3974_archive_v367_activate_v368.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "archive_v367_activate_v368_3974" in text


def test_req_report_3974_main_prints_written_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-3974: module main is a thin printable runner."""

    out_path = tmp_path / "results" / "experiment_3974_archive_v367_activate_v368.json"
    monkeypatch.setattr(mod, "run", lambda root: out_path)

    assert mod.main() == 0
    assert str(out_path) in capsys.readouterr().out
