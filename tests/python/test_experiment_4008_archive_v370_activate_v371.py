"""Tests for Exp 4008 .370 archive and .371 activation.

Spec refs: REQ-REPORT-4008, SCENARIO-REPORT-4008,
SCENARIO-REPORT-4008-BLOCKED-YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v370_activate_v371_4008 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")

SUMMARY_STDOUT = """\
!! no artifact matched: results/experiment_3997_archive_v369_activate_v370.json
==============================================================================
ARTIFACT  experiment_3998_gap4_deselection_coverage.json
------------------------------------------------------------------------------
  verdict          : complete: gap4_deselection_coverage_0.4091_n11
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 2475.8   substrate: codex_program_induction_executed_consistency_vs_cached_arc_pool
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3999_gap4_precision_confirmation_v2.json
------------------------------------------------------------------------------
  verdict          : complete: protocol_preregistered_pending_execution
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 1.0   substrate: codex_program_induction_all_fresh_k3_posthoc_arc2_gold_scoring
  headline metrics :
      precision_vs_fresharm_base = 0.0
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_4000_gap4_feedback_vs_redraw.json
------------------------------------------------------------------------------
  verdict          : complete: feedback_no_better_than_redraw_p1.0_FALSE_NEGATIVE_RISK
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 1902.1   substrate: codex_program_induction_same_run_feedback_vs_iid_redraw_arc2
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_4001_gap4_registration_offline_eval.json
------------------------------------------------------------------------------
  verdict          : success: gap4_stack_registered_arc2_19of31_arc1_28of31_reproduced
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.455   substrate: aggregation_from_upstream_artifacts
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_4002_gap4_local_generator_arm.json
------------------------------------------------------------------------------
  verdict          : complete: gap4_local_induction0.2581_pass20.4516_below_codex
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 1817.28   substrate: live_llm_inference
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_4003_scale_level_frontier.json
------------------------------------------------------------------------------
  verdict          : complete: level_frontier_holds_r11l_L4_no_verifier_validated_candidate_advanced_the_level_total5
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 7.099   substrate: offline_arc_agi3_gap4_executed_consistency_verifier_validated_frontier_scaling
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_4004_fourth_game_explore_first.json
------------------------------------------------------------------------------
  verdict          : success: fourth_game_solved_su15-1944f8ab_at_action14
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.56   substrate: offline_arc_agi3_explore_first_grounded_dynamics_gap4_pruner
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_4005_arcmemo_solve_transfer_v3.json
------------------------------------------------------------------------------
  verdict          : success: arcmemo_solve_transfer_v3_14to10_actions
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.4   substrate: offline_arc_agi3_real_env_steps_plus_gamegraph_arcmemo_concept_memory_v3
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_4006_hardware_continuity.json
------------------------------------------------------------------------------
  verdict          : complete: hardware_continuity_4006_kvreachable_overlay_absent_gmblocked_gatemate_unreachable_pfreachable_ssh_continuity_recorded
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 7.549209   substrate: hardware_smoke
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_4007_capstone_v370.json
------------------------------------------------------------------------------
  verdict          : success: capstone_v370_gap4_PHASE_RAN_UNCONFIRMED_DECENTRALIZED_DEPLOYED_local_not_beats_vote_games4_levels5_arcmemo_transfer_win_missing1_flagged_skipped0
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.33464160503353924   substrate: aggregation_from_upstream_artifacts
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


def _pretest_result(
    *,
    exit_code: int = 0,
    stdout: str = "all green\n",
) -> mod.CommandResult:
    return _command_result(mod.full_pretest_suite_command(), exit_code=exit_code, stdout=stdout)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _seed_repo(
    root: Path,
    *,
    corrupt_complete: bool = False,
    milestone: str = "2026.06.371",
    manifest: str = "retired: []\n",
    precision: dict[str, object] | None = None,
    deploy: dict[str, object] | None = None,
) -> None:
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{milestone}"\n'
        "tasks:\n"
        "  - id: exp4008-archive-v370-activate-v371\n",
        encoding="utf-8",
    )
    complete_text = (
        "milestones:\n"
        "- id: 2026.06.369\n"
        "  title: prior archive\n"
        "  completed: '2026-06-10'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3986-archive-v368-activate-v369\n"
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
    _write_json(
        root / "results" / "experiment_3999_gap4_precision_confirmation_v2.json",
        precision
        or {
            "honest_verdict": "complete: protocol_preregistered_pending_execution",
            "total_codex_calls": 0,
            "n_agreement_events": 0,
        },
    )
    _write_json(
        root / "results" / "experiment_4001_gap4_registration_offline_eval.json",
        deploy
        or {
            "honest_verdict": "success: gap4_stack_registered_arc2_19of31_arc1_28of31_reproduced",
            "verifier_registered": True,
            "arc2_reproduced_19of31": True,
            "arc1_reproduced_28of31": True,
        },
    )


def _run_success(root: Path, **overrides: object) -> Path:
    kwargs = {
        "research_complete_parse_result": _command_result(mod.research_complete_yaml_command()),
        "summary_result": _summary_result(),
        "arc_modules_import_result": _import_result(),
        "pretest_suite_results": [_pretest_result()],
        "started_s": 1.0,
        "now_s": 3.0,
    }
    kwargs.update(overrides)
    return mod.run(root, **kwargs)


def test_req_report_4008_spec_anchor_exists() -> None:
    """REQ-REPORT-4008: OpenSpec declares the .370 archive contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4008" in spec
    assert "SCENARIO-REPORT-4008" in spec
    assert "SCENARIO-REPORT-4008-BLOCKED-YAML" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "confirmation_still_owed_recorded" in spec
    assert "gap4_deployed_recorded" in spec


def test_scenario_report_4008_appends_truth_and_green_gate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4008: archive .370 and keep the anti-cascade gate."""

    _seed_repo(tmp_path)
    before = {
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
    assert artifact["honest_verdict"].startswith(("complete:", "success:"))
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert not isinstance(artifact[field], dict), field
    assert artifact["archived_milestone"] == "2026.06.370"
    assert artifact["activated_milestone"] == "2026.06.371"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["arc_modules_importable"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["quarantined_tests"] == []
    assert artifact["confirmation_still_owed_recorded"] is True
    assert artifact["gap4_deployed_recorded"] is True
    assert artifact["active_milestone_confirmed"] is True
    assert artifact["n_tasks_archived"] == len(mod.V370_TASKS)
    assert artifact["duration_s"] > 0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert "GGUF" not in artifact["honest_verdict"]
    assert "CUDA" not in artifact["inference_substrate"]

    assert complete_text.count(mod.ARCHIVE_MARKER) == 1
    assert "result: complete:" not in complete_text
    assert "result: 'complete:" in complete_text
    assert archived["activation_recorded"] == "exp4008-archive-v370-activate-v371"
    assert "missing_artifact:" in task_results["exp3997-archive-v369-activate-v370"]
    assert "protocol_preregistered_pending_execution" in task_results[
        "exp3999-gap4-precision-confirmation-v2"
    ]
    assert "gap4_stack_registered_arc2_19of31_arc1_28of31" in task_results[
        "exp4001-gap4-registration-offline-eval"
    ]
    assert yaml.safe_load(complete_text)

    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8") == before["manifest"]
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before["roadmap"]
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before["changelog"]
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before["status"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8") == before["trace"]
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(encoding="utf-8") == before["conductor"]


def test_req_report_4008_append_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-4008: rerunning does not duplicate the archive entry."""

    _seed_repo(tmp_path)

    first = _run_success(tmp_path).read_text(encoding="utf-8")
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second = _run_success(tmp_path).read_text(encoding="utf-8")

    assert first == second
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == first_complete
    assert first_complete.count(mod.ARCHIVE_MARKER) == 1


def test_scenario_report_4008_blocked_yaml_writes_artifact_without_edits(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4008-BLOCKED-YAML: corrupt YAML exits before edits."""

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
        arc_modules_import_result=_import_result(),
        pretest_suite_results=[_pretest_result()],
        started_s=7.0,
        now_s=7.5,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison")
    assert artifact["research_complete_yaml_parses"] is False
    assert artifact["exclusion_manifest_parses"] is False
    assert artifact["arc_modules_importable"] is False
    assert artifact["pretest_suite_green"] is False
    assert artifact["quarantined_tests"] == []
    assert artifact["confirmation_still_owed_recorded"] is False
    assert artifact["gap4_deployed_recorded"] is False
    assert artifact["preconditions_checked"]["research_complete_yaml_exists"] is True
    assert artifact["preconditions_checked"]["research_complete_yaml_parsed_before"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8") == before_manifest


def test_req_report_4008_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-4008: missing or stale handoff facts block instead of fabricating success."""

    _seed_repo(tmp_path, milestone="2026.06.370")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_v371_not_active")

    _seed_repo(tmp_path, manifest="retired: [\n")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_exclusion_manifest_yaml_poison")

    _seed_repo(tmp_path)
    artifact = json.loads(
        _run_success(tmp_path, summary_result=_summary_result(exit_code=127, stdout="")).read_text(
            encoding="utf-8"
        )
    )
    assert artifact["honest_verdict"].startswith("blocked_v370_summary_command_failed")

    _seed_repo(tmp_path, precision={"honest_verdict": "complete: pending", "total_codex_calls": 1, "n_agreement_events": 0})
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_confirmation_owed_record_missing")

    _seed_repo(tmp_path, deploy={"verifier_registered": True, "arc2_reproduced_19of31": True})
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_gap4_deploy_record_missing")

    _seed_repo(tmp_path)
    artifact = json.loads(
        _run_success(tmp_path, arc_modules_import_result=_import_result(all_ok=False)).read_text(
            encoding="utf-8"
        )
    )
    assert artifact["honest_verdict"].startswith("blocked_arc_module_import")


def test_req_report_4008_quarantines_red_pretest_files(tmp_path: Path) -> None:
    """REQ-REPORT-4008: red full-suite files are moved out of tests/python."""

    _seed_repo(tmp_path)
    red_file = tmp_path / "tests" / "python" / "test_old_poison.py"
    red_file.parent.mkdir(parents=True, exist_ok=True)
    red_file.write_text("def test_bad():\n    assert False\n", encoding="utf-8")
    failure = _pretest_result(
        exit_code=1,
        stdout="FAILED tests/python/test_old_poison.py::test_bad - AssertionError\n",
    )

    artifact = json.loads(
        _run_success(
            tmp_path,
            pretest_suite_results=[failure, _pretest_result()],
        ).read_text(encoding="utf-8")
    )

    assert artifact["pretest_suite_green"] is True
    assert artifact["quarantined_tests"] == [
        {
            "path": "tests/python/test_old_poison.py",
            "quarantined_path": "tests/quarantine/test_old_poison.py",
            "failing_test_ids": ["tests/python/test_old_poison.py::test_bad"],
        }
    ]
    assert not red_file.exists()
    assert (tmp_path / "tests" / "quarantine" / "test_old_poison.py").exists()
    assert (tmp_path / "tests" / "quarantine" / "__init__.py").exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("pretest_suite_green"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p.update(archived_milestone="2026.06.369"), "archived milestone"),
        (lambda p: p.update(activated_milestone="2026.06.370"), "activated milestone"),
        (lambda p: p.update(research_complete_yaml_parses=False), "research-complete"),
        (lambda p: p.update(exclusion_manifest_parses=False), "manifest"),
        (lambda p: p.update(arc_modules_importable=False), "ARC module imports"),
        (lambda p: p.update(pretest_suite_green=False), "pretest suite"),
        (lambda p: p.update(confirmation_still_owed_recorded=False), "confirmation"),
        (lambda p: p.update(gap4_deployed_recorded=False), "deploy"),
        (lambda p: p.update(active_milestone_confirmed=False), "active milestone"),
        (lambda p: p.update(n_tasks_archived=3), "n_tasks_archived"),
        (lambda p: p.update(inference_substrate="live_model"), "inference"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(quarantined_tests={}), "quarantined_tests"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(copied_marker="CUDA"), "compute-bound markers"),
    ],
)
def test_req_report_4008_validate_artifact_rejects_regressions(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-4008: validation rejects fields that hide transition risk."""

    _seed_repo(tmp_path)
    payload = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))

    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(broken)


def test_req_report_4008_summary_import_and_quarantine_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-4008: helpers preserve stable evidence and fail closed."""

    _seed_repo(tmp_path)
    records = mod.parse_summary_records(SUMMARY_STDOUT)
    verdicts = mod.task_verdicts_from_summary(SUMMARY_STDOUT)
    summary = mod.build_prior_verdicts_summary(verdicts)

    assert records["4002"]["duration_s"] == pytest.approx(1817.28)
    assert "missing_artifact:" in verdicts["exp3997-archive-v369-activate-v370"]
    assert "exp4007: success: capstone_v370" in summary
    assert mod.confirmation_still_owed_from_file(
        tmp_path / "results" / "experiment_3999_gap4_precision_confirmation_v2.json"
    ) is True
    assert mod.gap4_deployed_from_file(
        tmp_path / "results" / "experiment_4001_gap4_registration_offline_eval.json"
    ) is True
    assert mod.confirmation_still_owed_from_file(tmp_path / "missing.json") is False
    assert mod.gap4_deployed_from_file(tmp_path / "missing.json") is False
    assert mod.yaml_single_quote("complete: ok") == "'complete: ok'"
    assert mod.duration_from(None, None) == 0.0001
    assert mod._milestone_from_text("tasks: []\n") == "unknown"
    assert mod.parse_failing_test_ids("FAILED tests/python/a.py::test_x - err\n") == {
        "tests/python/a.py": ["tests/python/a.py::test_x"]
    }
    assert mod.parse_failing_test_ids("no failures") == {}
    assert mod._artifact_key_from_line("ARTIFACT  arc3_gap4_rule_exec_verifier.json") == (
        "arc3_gap4_rule_exec_verifier"
    )
    bad_duration = SUMMARY_STDOUT.replace("duration_s       : 1817.28", "duration_s       : None")
    assert mod.parse_summary_records(bad_duration)["4002"]["duration_s"] is None

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


def test_req_report_4008_edge_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-4008: defensive fallback paths stay explicit and covered."""

    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    assert mod._dedup_paths(["a", "b", "a"]) == ["a", "b"]

    def _mutate_4001_block(summary: str, old: str, new: str) -> str:
        marker = "ARTIFACT  experiment_4001_gap4_registration_offline_eval.json"
        head, sep, tail = summary.partition(marker)
        assert sep, "exp4001 record missing from fixture"
        return head + sep + tail.replace(old, new, 1)

    critical_summary = _mutate_4001_block(
        SUMMARY_STDOUT, "LIVE re-check: clean", "LIVE re-check: CRITICAL"
    )
    critical_verdicts = mod.task_verdicts_from_summary(critical_summary)
    assert "LIVE_CRITICAL" in critical_verdicts["exp4001-gap4-registration-offline-eval"]

    flagged_summary = _mutate_4001_block(
        SUMMARY_STDOUT,
        "flagged_adversarial (stamped): None",
        "flagged_adversarial (stamped): True",
    )
    flagged_verdicts = mod.task_verdicts_from_summary(flagged_summary)
    assert "stamped_flagged" in flagged_verdicts["exp4001-gap4-registration-offline-eval"]

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").unlink()
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison_missing")

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
    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison_after_append")
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

    _seed_repo(tmp_path)
    red_without_ids = _pretest_result(exit_code=1, stdout="assertion failed without node id")
    artifact = json.loads(
        _run_success(tmp_path, pretest_suite_results=[red_without_ids]).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_pretest_suite_failed_unquarantined")
    assert artifact["pretest_suite_green"] is False

    green = _pretest_result()
    monkeypatch.setattr(mod, "run_full_pretest_suite", lambda root: green)
    assert mod.run_pretest_until_green(tmp_path, supplied=None) == (True, [], [green])

    repeated_failure = _pretest_result(
        exit_code=1,
        stdout="FAILED tests/python/test_missing.py::test_bad - AssertionError\n",
    )
    monkeypatch.setattr(mod, "quarantine_failed_tests", lambda root, failures: [])
    assert mod.run_pretest_until_green(tmp_path, supplied=[repeated_failure] * 8) == (
        False,
        [],
        [repeated_failure] * 8,
    )


def test_req_report_4008_subprocess_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-4008: subprocess helpers use the mandated commands."""

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
    assert mod.run_arc_modules_import_check(tmp_path).stdout == _import_stdout()
    assert calls == [mod.arc_modules_import_command()]

    calls.clear()
    assert mod.run_full_pretest_suite(tmp_path).stdout == _import_stdout()
    assert calls == [mod.full_pretest_suite_command()]

    def os_error_subprocess(cmd: list[str], **kwargs: object) -> object:
        raise OSError("missing executable")

    monkeypatch.setattr(mod.subprocess, "run", os_error_subprocess)
    assert mod.run_research_complete_parse_check(tmp_path).exit_code == 127
    assert mod.run_summarize_artifacts(tmp_path).exit_code == 127
    assert mod.run_full_pretest_suite(tmp_path).exit_code == 127


def test_scenario_report_4008_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-4008: the requested experiment entrypoint exists."""

    script = Path("scripts/experiments/experiment_4008_archive_v370_activate_v371.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "archive_v370_activate_v371_4008" in text
