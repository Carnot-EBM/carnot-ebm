"""Tests for Exp 3986 .368 archive and .369 activation.

Spec refs: REQ-REPORT-3986, SCENARIO-REPORT-3986,
SCENARIO-REPORT-3986-BLOCKED-YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v368_activate_v369_3986 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")

SUMMARY_STDOUT = """\
==============================================================================
ARTIFACT  arc3_gap4_rule_exec_verifier.json
------------------------------------------------------------------------------
  verdict          : complete: gap4_rule_exec_BEATS_vote_n31_vote_0.4516_gated_0.5806_recovered_4_lost_0_demoperfect_29of31
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 374.6   substrate: codex_program_induction_plus_offline_trm_candidate_rerank_no_oracle
  adversarial flags: none
==============================================================================
ARTIFACT  arc3_gap4_chain_arms_adversarial_verify.json
------------------------------------------------------------------------------
  verdict          : complete: gap4_chain_arms_confirmed_prereg_honestly_failed_coverage_lift_real_precision_uplift_not_established
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : None   substrate: aggregation_from_upstream_artifacts
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3974_archive_v367_activate_v368.json
------------------------------------------------------------------------------
  verdict          : complete: archived_v367_v368_active_three_games_m3_open_gap4_ready_arc_substrate_green
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 13.148285   substrate: aggregation_from_upstream_artifacts
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3975_gap4_execution_verifier_build.json
------------------------------------------------------------------------------
  verdict          : complete: gap4_positive_control_failed_auroc0.00
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.07524847984313965   substrate: dsl-only
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3976_gap4_trm_rerank_eval.json
------------------------------------------------------------------------------
  verdict          : blocked_gate_check_failed
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.0   substrate: None
  adversarial flags: none
!! no artifact matched: results/experiment_3977_gap4_rederive_audit.json
==============================================================================
ARTIFACT  experiment_3978_verifier_vs_judge_efficiency.json
------------------------------------------------------------------------------
  verdict          : success: verifier_earns_place_efficiency_parity_8789.7x_cheaper
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 31.8   substrate: offline_arc_agi3_plus_local_judge
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3979_world_model_gen_execution_guided.json
------------------------------------------------------------------------------
  verdict          : complete: exec_guided_trustworthy_0of6
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 61.8   substrate: offline_arc_agi3_execution_guided_program_synthesis_exact_replay_consistency_verified
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3980_incremental_levels_reinduction.json
------------------------------------------------------------------------------
  verdict          : complete: l2_wall_holds_r11l_l2_re_induction_found_a_collision_forbidden_mask_rule
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.678   substrate: offline_arc_agi3_per_level_execution_guided_reinduction
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3981_fourth_game_first_solve.json
------------------------------------------------------------------------------
  verdict          : complete: fourth_game_no_solve_budget_exceeded
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.0   substrate: offline_arc_agi3_perception_planner_real_env_confirmed
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3982_arcmemo_solve_transfer.json
------------------------------------------------------------------------------
  verdict          : success: arcmemo_solve_transfer_2668to17_actions
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 1.9   substrate: offline_arc_agi3_real_env_steps_plus_gamegraph_arcmemo_concept_memory
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3983_hardware_continuity.json
------------------------------------------------------------------------------
  verdict          : complete: hardware_continuity_3983_kvreachable_overlay_absent_gmblocked_gatemate_unreachable_pfreachable_ssh_continuity_recorded
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 7.555079   substrate: hardware_smoke
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3984_retro_commit_detector_fix.json
------------------------------------------------------------------------------
  verdict          : complete: retro_commit_detector_fixed_backfill_counts_restored
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.342871   substrate: git_history_added_terminal_artifact_scan
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3985_capstone_v368.json
------------------------------------------------------------------------------
  verdict          : success: capstone_v368_verifier_earned_efficiency_only_games3_new_levels0_missing1_flagged_skipped0
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.3841149849467911   substrate: aggregation_from_upstream_artifacts_via_summarize_artifact_py
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


def _write_special_evidence(
    root: Path,
    *,
    followups: list[str] | None = None,
    positive_verdict: str = (
        "complete: gap4_rule_exec_BEATS_vote_n31_vote_0.4516_gated_0.5806"
    ),
) -> None:
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "arc3_gap4_rule_exec_verifier.json").write_text(
        json.dumps({"honest_verdict": positive_verdict}, sort_keys=True),
        encoding="utf-8",
    )
    (results / "arc3_gap4_chain_arms_adversarial_verify.json").write_text(
        json.dumps(
            {"synthesis": {"conductor_followups": followups or ["a", "b", "c", "d"]}},
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _seed_repo(
    root: Path,
    *,
    corrupt_complete: bool = False,
    milestone: str = "2026.06.369",
    manifest: str = "retired: []\n",
    followups: list[str] | None = None,
) -> None:
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{milestone}"\n'
        "tasks:\n"
        "  - id: exp3986-archive-v368-activate-v369\n",
        encoding="utf-8",
    )
    complete_text = (
        "milestones:\n"
        "- id: 2026.06.368\n"
        "  title: conductor placeholder\n"
        "  completed: '2026-06-10'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3975-gap4-execution-verifier-build\n"
        "    result: OK (conductor)\n"
    )
    if corrupt_complete:
        complete_text += "  - id: poison\n    result: complete: unquoted colon\n"
    (root / "research-complete.yaml").write_text(complete_text, encoding="utf-8")
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "experiments").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(manifest, encoding="utf-8")
    (root / "ops" / "changelog.md").write_text("changelog before\n", encoding="utf-8")
    (root / "ops" / "status.md").write_text("status before\n", encoding="utf-8")
    (root / "_bmad" / "traceability.md").write_text("trace before\n", encoding="utf-8")
    (root / "scripts" / "research_conductor.py").write_text("# conductor before\n", encoding="utf-8")
    _write_special_evidence(root, followups=followups)


def _run_success(root: Path, **overrides: object) -> Path:
    kwargs = {
        "research_complete_parse_result": _command_result(
            mod.research_complete_yaml_command()
        ),
        "summary_result": _summary_result(),
        "arc_substrate_test_result": _command_result(mod.arc_substrate_test_command()),
        "arc_modules_import_result": _import_result(),
        "started_s": 1.0,
        "now_s": 3.5,
    }
    kwargs.update(overrides)
    return mod.run(root, **kwargs)


def test_req_report_3986_spec_anchor_exists() -> None:
    """REQ-REPORT-3986: OpenSpec declares the .368 archive contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3986" in spec
    assert "SCENARIO-REPORT-3986" in spec
    assert "SCENARIO-REPORT-3986-BLOCKED-YAML" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "gap4_outer_loop_positive_recorded" in spec
    assert "conductor_dsl_build_failed_recorded" in spec
    assert "followups_present" in spec


def test_scenario_report_3986_run_appends_truth_and_green_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3986: archive .368 and activate .369."""

    _seed_repo(tmp_path)
    before = {
        "manifest": (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(
            encoding="utf-8"
        ),
        "roadmap": (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8"),
        "changelog": (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8"),
        "status": (tmp_path / "ops" / "status.md").read_text(encoding="utf-8"),
        "trace": (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8"),
        "conductor": (tmp_path / "scripts" / "research_conductor.py").read_text(
            encoding="utf-8"
        ),
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
    assert artifact["archived_milestone"] == "2026.06.368"
    assert artifact["activated_milestone"] == "2026.06.369"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["arc_substrate_tests_green"] is True
    assert artifact["arc_modules_importable"] is True
    assert artifact["gap4_outer_loop_positive_recorded"] is True
    assert artifact["conductor_dsl_build_failed_recorded"] is True
    assert artifact["followups_present"] is True
    assert artifact["active_milestone_confirmed"] is True
    assert artifact["n_tasks_archived"] == 12
    assert artifact["duration_s"] == 2.5
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert "GGUF" not in artifact["honest_verdict"]
    assert "CUDA" not in artifact["inference_substrate"]

    assert complete_text.count(mod.ARCHIVE_MARKER) == 1
    assert complete_text.count("- id: 2026.06.368") == 2
    assert "result: complete:" not in complete_text
    assert "result: 'complete:" in complete_text
    assert archived["activation_recorded"] == "exp3986-archive-v368-activate-v369"
    assert "gap4_positive_control_failed_auroc0.00" in task_results[
        "exp3975-gap4-execution-verifier-build"
    ]
    assert "missing_artifact:" in task_results["exp3977-gap4-rederive-audit"]
    assert "arcmemo_solve_transfer_2668to17_actions" in task_results[
        "exp3982-arcmemo-solve-transfer"
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


def test_req_report_3986_append_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3986: rerunning does not duplicate the archive entry."""

    _seed_repo(tmp_path)

    first = _run_success(tmp_path).read_text(encoding="utf-8")
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second = _run_success(tmp_path).read_text(encoding="utf-8")

    assert first == second
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == first_complete
    assert first_complete.count(mod.ARCHIVE_MARKER) == 1


def test_scenario_report_3986_blocked_yaml_writes_artifact_without_edits(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3986-BLOCKED-YAML: corrupt YAML exits before edits."""

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
    assert artifact["gap4_outer_loop_positive_recorded"] is False
    assert artifact["conductor_dsl_build_failed_recorded"] is False
    assert artifact["followups_present"] is False
    assert artifact["preconditions_checked"]["research_complete_yaml_exists"] is True
    assert artifact["preconditions_checked"]["research_complete_yaml_parsed_before"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8") == before_manifest


def test_req_report_3986_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3986: hard helper failures block instead of fabricating success."""

    _seed_repo(tmp_path, milestone="2026.06.368")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_v369_not_active")

    _seed_repo(tmp_path, manifest="retired: [\n")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_exclusion_manifest_yaml_poison")

    _seed_repo(tmp_path, followups=["only one"])
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_followups_missing")

    _seed_repo(tmp_path)
    no_positive = SUMMARY_STDOUT.replace("gap4_rule_exec_BEATS_vote", "gap4_rule_exec_no_lift")
    artifact = json.loads(
        _run_success(tmp_path, summary_result=_summary_result(stdout=no_positive)).read_text(
            encoding="utf-8"
        )
    )
    assert artifact["honest_verdict"].startswith("blocked_gap4_outer_loop_positive_missing")

    _seed_repo(tmp_path)
    no_dsl_fail = SUMMARY_STDOUT.replace("gap4_positive_control_failed", "gap4_positive_control_passed")
    artifact = json.loads(
        _run_success(tmp_path, summary_result=_summary_result(stdout=no_dsl_fail)).read_text(
            encoding="utf-8"
        )
    )
    assert artifact["honest_verdict"].startswith("blocked_conductor_dsl_failure_missing")

    _seed_repo(tmp_path)
    artifact = json.loads(
        _run_success(tmp_path, summary_result=_summary_result(exit_code=127, stdout="")).read_text(
            encoding="utf-8"
        )
    )
    assert artifact["honest_verdict"].startswith("blocked_v368_summary_command_failed")

    _seed_repo(tmp_path)
    artifact = json.loads(
        _run_success(
            tmp_path,
            arc_substrate_test_result=_command_result(mod.arc_substrate_test_command(), exit_code=1),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_arc_substrate_tests_failed")

    _seed_repo(tmp_path)
    artifact = json.loads(
        _run_success(tmp_path, arc_modules_import_result=_import_result(all_ok=False)).read_text(
            encoding="utf-8"
        )
    )
    assert artifact["honest_verdict"].startswith("blocked_arc_module_import")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("followups_present"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p.update(archived_milestone="2026.06.367"), "archived milestone"),
        (lambda p: p.update(activated_milestone="2026.06.368"), "activated milestone"),
        (lambda p: p.update(research_complete_yaml_parses=False), "research-complete"),
        (lambda p: p.update(exclusion_manifest_parses=False), "manifest"),
        (lambda p: p.update(arc_substrate_tests_green=False), "ARC substrate"),
        (lambda p: p.update(arc_modules_importable=False), "ARC module imports"),
        (lambda p: p.update(gap4_outer_loop_positive_recorded=False), "outer-loop"),
        (lambda p: p.update(conductor_dsl_build_failed_recorded=False), "DSL"),
        (lambda p: p.update(followups_present=False), "followups"),
        (lambda p: p.update(active_milestone_confirmed=False), "active milestone"),
        (lambda p: p.update(n_tasks_archived=11), "n_tasks_archived"),
        (lambda p: p.update(inference_substrate="live_model"), "inference"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(copied_marker="CUDA"), "compute-bound markers"),
    ],
)
def test_req_report_3986_validate_artifact_rejects_regressions(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3986: validation rejects fields that hide transition risk."""

    _seed_repo(tmp_path)
    payload = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))

    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(broken)


def test_req_report_3986_summary_followup_and_import_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-3986: helpers preserve evidence and fail closed."""

    _seed_repo(tmp_path)
    records = mod.parse_summary_records(SUMMARY_STDOUT)
    verdicts = mod.task_verdicts_from_summary(SUMMARY_STDOUT)
    summary = mod.build_prior_verdicts_summary(verdicts)

    assert records["3978"]["duration_s"] == pytest.approx(31.8)
    assert "gap4_positive_control_failed" in verdicts[
        "exp3975-gap4-execution-verifier-build"
    ]
    assert "missing_artifact:" in verdicts["exp3977-gap4-rederive-audit"]
    assert "exp3985: success: capstone_v368" in summary
    assert mod.gap4_outer_loop_positive_from_summary(SUMMARY_STDOUT) is True
    assert mod.conductor_dsl_failure_from_verdicts(verdicts) is True
    assert mod.followups_present_from_file(
        tmp_path / "results" / "arc3_gap4_chain_arms_adversarial_verify.json"
    ) is True
    assert mod.followups_present_from_file(tmp_path / "missing.json") is False
    assert mod.yaml_single_quote("complete: ok") == "'complete: ok'"
    assert mod.duration_from(None, None) == 0.0001
    assert mod._milestone_from_text("tasks: []\n") == "unknown"

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


def test_req_report_3986_edge_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3986: defensive fallback paths stay explicit and covered."""

    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    assert mod._dedup_paths(["a", "b", "a"]) == ["a", "b"]

    # Scope the mutation to the exp3974 record: a bare .replace(..., 1) hits the FIRST record in
    # the fixture (the gap4 supplementary artifact), not exp3974, so the original assertions tested
    # an unmodified record (2026-06-10 pre-test poison fix — wrong-target replace, impl was correct).
    def _mutate_3974_block(summary: str, old: str, new: str) -> str:
        marker = "ARTIFACT  experiment_3974_archive_v367_activate_v368.json"
        head, sep, tail = summary.partition(marker)
        assert sep, "exp3974 record missing from fixture"
        return head + sep + tail.replace(old, new, 1)

    critical_summary = _mutate_3974_block(
        SUMMARY_STDOUT, "LIVE re-check: clean", "LIVE re-check: CRITICAL"
    )
    critical_verdicts = mod.task_verdicts_from_summary(critical_summary)
    assert "LIVE_CRITICAL" in critical_verdicts["exp3974-archive-v367-activate-v368"]

    flagged_summary = _mutate_3974_block(
        SUMMARY_STDOUT,
        "flagged_adversarial (stamped): None",
        "flagged_adversarial (stamped): True",
    )
    flagged_verdicts = mod.task_verdicts_from_summary(flagged_summary)
    assert "stamped_flagged" in flagged_verdicts["exp3974-archive-v367-activate-v368"]

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").unlink()
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith(
        "blocked_research_complete_yaml_poison_missing"
    )

    # monkeypatch undoes at test END, not block end — capture the real functions so each
    # fault-injection block can restore them before the next _run_success (2026-06-10 pre-test
    # poison fix: the exhausted parses_sequence iterator leaked into later runs -> StopIteration).
    real_append = mod.append_research_complete_record
    real_yaml_parses = mod.yaml_parses

    _seed_repo(tmp_path)
    monkeypatch.setattr(mod, "append_research_complete_record", lambda *args: "milestones: [\n")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_append_invalid")
    monkeypatch.setattr(mod, "append_research_complete_record", real_append)

    _seed_repo(tmp_path)
    states = iter([True, True, False, True])

    def parses_sequence(text: str) -> bool:
        return next(states)

    monkeypatch.setattr(mod, "yaml_parses", parses_sequence)
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith(
        "blocked_research_complete_yaml_poison_after_append"
    )
    monkeypatch.setattr(mod, "yaml_parses", real_yaml_parses)

    _seed_repo(tmp_path)
    payload = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    bad = dict(payload, honest_verdict="maybe")
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(payload))
    bad["field_principles"].pop("duration_s")
    with pytest.raises(ValueError, match="missing field principles"):
        mod.validate_artifact(bad)


def test_req_report_3986_subprocess_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3986: subprocess helpers use the mandated commands."""

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

    def os_error_subprocess(cmd: list[str], **kwargs: object) -> object:
        raise OSError("missing executable")

    monkeypatch.setattr(mod.subprocess, "run", os_error_subprocess)
    assert mod.run_research_complete_parse_check(tmp_path).exit_code == 127
    assert mod.run_summarize_artifacts(tmp_path).exit_code == 127


def test_scenario_report_3986_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3986: the requested experiment entrypoint exists."""

    script = Path("scripts/experiments/experiment_3986_archive_v368_activate_v369.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "archive_v368_activate_v369_3986" in text
