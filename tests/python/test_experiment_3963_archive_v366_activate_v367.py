"""Tests for Exp 3963 .366 archive and .367 activation.

Spec refs: REQ-REPORT-3963, SCENARIO-REPORT-3963,
SCENARIO-REPORT-3963-BLOCKED-YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v366_activate_v367_3963 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


SUMMARY_STDOUT = """\
!! no artifact matched: results/experiment_3953_r11l_full_solve.json
!! no artifact matched: results/experiment_3955_active_codex_nonspatial_sweep.json
!! no artifact matched: results/experiment_3960_m4_offline_accuracy_sweep.json
!! no artifact matched: results/experiment_3961_hardware_continuity.json
!! no artifact matched: results/experiment_3962_capstone_v366.json
==============================================================================
ARTIFACT  experiment_3952_archive_v365_activate_v366.json
------------------------------------------------------------------------------
  verdict          : complete: archived_v365_v366_active_first_solve_recorded_arc_substrate_green
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 11.056669   substrate: aggregation_from_upstream_artifacts
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3954_second_game_solve.json
------------------------------------------------------------------------------
  verdict          : complete: second_game_solve_lp85-305b61c3_levels1_solvedTrue
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.4   substrate: offline_arc_agi3_perception_planner_real_env_confirmed
  headline metrics :
      ACCURACY_levels_solved = 1
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3956_goal_predicate_induction.json
------------------------------------------------------------------------------
  verdict          : complete: precision_0.40_recall_1.00
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.81   substrate: offline_arc_agi3_heuristic_inducer
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3957_hidden_state_latent_registers.json
------------------------------------------------------------------------------
  verdict          : complete: latent_registers_no_drop_energy
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 25.650296926498413   substrate: offline_arc_agi3_latent_registers
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3958_cross_game_dsl_transfer.json
------------------------------------------------------------------------------
  verdict          : complete: dsl_transfer_no_win_no_cost_reduction_observed
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 823.2   substrate: offline_arc_agi3_plus_codex_program_synthesis_consistency_verified
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3959_m3_efficiency_real_games.json
------------------------------------------------------------------------------
  verdict          : complete: m3_efficiency_real_games_pruner_helps
  flagged_adversarial (stamped): True   |   LIVE re-check: clean
  duration_s       : 0.9   substrate: offline_air_gapped_arc_agi3_local_environments
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
    milestone: str = "2026.06.367",
    manifest: str | None = None,
) -> None:
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{milestone}"\n'
        "tasks:\n"
        "  - id: exp3963-archive-v366-activate-v367\n",
        encoding="utf-8",
    )
    complete_text = (
        "milestones:\n"
        "- id: 2026.06.366\n"
        "  title: stale conductor archive\n"
        "  completed: '2026-06-09'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3954-second-game-solve\n"
        "    deliverable: results/experiment_3954_second_game_solve.json\n"
        "    result: OK (conductor)\n"
        "  - id: exp3959-m3-efficiency-real-games\n"
        "    deliverable: results/experiment_3959_m3_efficiency_real_games.json\n"
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
        manifest or "retired_experiments:\n  - experiment_id: 3959\n    reason: existing\n",
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
        "now_s": 2.5,
    }
    kwargs.update(overrides)
    return mod.run(root, **kwargs)


def test_req_report_3963_spec_anchor_exists() -> None:
    """REQ-REPORT-3963: OpenSpec declares the .366 archive contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3963" in spec
    assert "SCENARIO-REPORT-3963" in spec
    assert "SCENARIO-REPORT-3963-BLOCKED-YAML" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "prior_milestone_second_solve_recorded" in spec
    assert "prior_m3_fabrication_recorded" in spec


def test_scenario_report_3963_run_appends_truth_and_green_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3963: archive .366 and preserve the recorded truth."""

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
    assert artifact["archived_milestone"] == "2026.06.366"
    assert artifact["activated_milestone"] == "2026.06.367"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["arc_substrate_tests_green"] is True
    assert artifact["arc_modules_importable"] is True
    assert artifact["prior_milestone_second_solve_recorded"] is True
    assert artifact["prior_m3_fabrication_recorded"] is True
    assert artifact["n_tasks_archived"] == 11
    assert artifact["duration_s"] == 1.5
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert "exp3954: complete: second_game_solve_lp85-305b61c3" in artifact[
        "prior_milestone_verdicts_summary"
    ]
    assert "exp3959: complete: m3_efficiency_real_games_pruner_helps" in artifact[
        "prior_milestone_verdicts_summary"
    ]
    assert "VERIFIER_NOT_IN_LOOP" in artifact["prior_milestone_verdicts_summary"]
    assert "exp3962: missing_artifact:" in artifact["prior_milestone_verdicts_summary"]
    assert artifact["arc_module_import_results"]["carnot.agentic.arc_agi3_action_efficiency"][
        "import_ok"
    ] is True

    assert complete_text.startswith(before["complete"].rstrip())
    assert complete_text.count(mod.ARCHIVE_MARKER) == 1
    assert complete_text.count("- id: 2026.06.366") == 2
    assert "result: complete:" not in complete_text
    assert "result: 'complete:" in complete_text
    assert archived["activation_recorded"] == "exp3963-archive-v366-activate-v367"
    assert task_results["exp3954-second-game-solve"].startswith("complete: second_game_solve")
    assert "SECOND_SOLVE" in task_results["exp3954-second-game-solve"]
    assert task_results["exp3959-m3-efficiency-real-games"].startswith(
        "complete: m3_efficiency"
    )
    assert "M3_FABRICATION" in task_results["exp3959-m3-efficiency-real-games"]
    assert task_results["exp3953-r11l-full-solve"].startswith("missing_artifact:")
    assert task_results["exp3955-active-codex-nonspatial-sweep"].startswith("missing_artifact:")
    assert task_results["exp3960-m4-offline-accuracy-sweep"].startswith("missing_artifact:")
    assert task_results["exp3961-hardware-continuity"].startswith("missing_artifact:")
    assert task_results["exp3962-capstone-v366"].startswith("missing_artifact:")
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


def test_req_report_3963_append_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3963: rerunning does not duplicate the archive entry."""

    _seed_repo(tmp_path)

    first = _run_success(tmp_path).read_text(encoding="utf-8")
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second = _run_success(tmp_path).read_text(encoding="utf-8")

    assert first == second
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == first_complete
    assert first_complete.count(mod.ARCHIVE_MARKER) == 1


def test_scenario_report_3963_blocked_yaml_writes_artifact_without_edits(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3963-BLOCKED-YAML: corrupt YAML exits before edits."""

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
    assert artifact["prior_milestone_second_solve_recorded"] is False
    assert artifact["prior_m3_fabrication_recorded"] is False
    assert artifact["preconditions_checked"]["research_complete_yaml_exists"] is True
    assert artifact["preconditions_checked"]["research_complete_yaml_parsed_before"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8") == before_manifest


def test_req_report_3963_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3963: hard helper failures block instead of fabricating success."""

    _seed_repo(tmp_path, milestone="2026.06.366")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_v367_not_active")

    _seed_repo(tmp_path, manifest="retired: [\n")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_exclusion_manifest_yaml_poison")

    _seed_repo(tmp_path)
    (tmp_path / "ops" / "exclusion_manifest.yaml").unlink()
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_exclusion_manifest_missing")

    _seed_repo(tmp_path)
    artifact = json.loads(
        _run_success(tmp_path, summary_result=_summary_result(exit_code=127, stdout="")).read_text(
            encoding="utf-8"
        )
    )
    assert artifact["honest_verdict"].startswith("blocked_v366_summary_command_failed")

    _seed_repo(tmp_path)
    no_second_solve = SUMMARY_STDOUT.replace(
        "complete: second_game_solve_lp85-305b61c3_levels1_solvedTrue",
        "blocked_no_second_solve",
    )
    artifact = json.loads(
        _run_success(tmp_path, summary_result=_summary_result(stdout=no_second_solve)).read_text(
            encoding="utf-8"
        )
    )
    assert artifact["honest_verdict"].startswith("blocked_prior_milestone_second_solve_missing")

    _seed_repo(tmp_path)
    no_m3_flag = SUMMARY_STDOUT.replace("flagged_adversarial (stamped): True", "flagged_adversarial (stamped): None")
    artifact = json.loads(
        _run_success(tmp_path, summary_result=_summary_result(stdout=no_m3_flag)).read_text(
            encoding="utf-8"
        )
    )
    assert artifact["honest_verdict"].startswith("blocked_prior_m3_fabrication_missing")

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
        (lambda p: p.pop("prior_milestone_second_solve_recorded"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p.update(archived_milestone="2026.06.365"), "archived milestone"),
        (lambda p: p.update(activated_milestone="2026.06.366"), "activated milestone"),
        (lambda p: p.update(research_complete_yaml_parses=False), "research-complete"),
        (lambda p: p.update(exclusion_manifest_parses=False), "manifest"),
        (lambda p: p.update(arc_substrate_tests_green=False), "ARC substrate"),
        (lambda p: p.update(arc_modules_importable=False), "ARC module imports"),
        (lambda p: p.update(prior_milestone_second_solve_recorded=False), "second solve"),
        (lambda p: p.update(prior_m3_fabrication_recorded=False), "M3 fabrication"),
        (lambda p: p.update(active_milestone_confirmed=False), "active milestone"),
        (lambda p: p.update(n_tasks_archived=10), "n_tasks_archived"),
        (lambda p: p.update(prior_milestone_verdicts_summary="exp3952: ok"), "missing exp3953"),
        (lambda p: p.update(inference_substrate="live_model"), "inference"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(copied_marker="CUDA"), "compute-bound markers"),
    ],
)
def test_req_report_3963_validate_artifact_rejects_regressions(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3963: validation rejects fields that hide transition risk."""

    _seed_repo(tmp_path)
    payload = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))

    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(broken)


def test_req_report_3963_summary_and_import_helpers() -> None:
    """REQ-REPORT-3963: summaries preserve missing tasks, solve truth, and imports."""

    records = mod.parse_summary_records(SUMMARY_STDOUT)
    verdicts = mod.task_verdicts_from_summary(SUMMARY_STDOUT)
    summary = mod.build_prior_verdicts_summary(verdicts)
    critical_summary = SUMMARY_STDOUT.replace("LIVE re-check: clean", "LIVE re-check: CRITICAL", 1)
    critical_verdicts = mod.task_verdicts_from_summary(critical_summary)

    assert records["3954"]["duration_s"] == pytest.approx(0.4)
    assert verdicts["exp3953-r11l-full-solve"].startswith("missing_artifact:")
    assert "SECOND_SOLVE" in verdicts["exp3954-second-game-solve"]
    assert "M3_FABRICATION" in verdicts["exp3959-m3-efficiency-real-games"]
    assert "LIVE_CRITICAL" in critical_verdicts["exp3952-archive-v365-activate-v366"]
    assert "exp3962: missing_artifact" in summary
    assert mod.yaml_single_quote("complete: ok") == "'complete: ok'"
    assert mod.duration_from(None, None) == 0.0001
    assert mod._milestone_from_text("tasks: []\n") == "unknown"
    assert mod.second_solve_recorded_from_text("milestones: [\n") is False
    assert mod.second_solve_recorded_from_text("milestones:\n- id: 2026.06.365\n") is False
    assert mod.m3_fabrication_recorded_from_text("tasks: []\n") is False
    assert mod.m3_fabrication_recorded_from_text("milestones:\n- id: 2026.06.366\n  tasks: nope\n") is False
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


def test_req_report_3963_edge_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3963: defensive fallback paths stay explicit and covered."""

    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    odd_summary = """\
==============================================================================
ARTIFACT  experiment_3959_m3_efficiency_real_games.json
------------------------------------------------------------------------------
  verdict          : complete: m3
  flagged_adversarial (stamped): True   |   LIVE re-check: warn
  duration_s       : not-a-number   substrate: aggregation
  adversarial flags: none
==============================================================================
"""
    records = mod.parse_summary_records(odd_summary)
    verdicts = mod.task_verdicts_from_summary(odd_summary)
    assert records["3959"]["duration_s"] is None
    assert "M3_FABRICATION" in verdicts["exp3959-m3-efficiency-real-games"]

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").unlink()
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison_missing")

    _seed_repo(tmp_path)
    monkeypatch.setattr(mod, "append_research_complete_record", lambda *args: "milestones: [\n")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_append_invalid")


def test_req_report_3963_subprocess_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3963: subprocess helpers use the mandated commands."""

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


def test_scenario_report_3963_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3963: the requested script entrypoint exists."""

    script = Path("scripts/experiments/experiment_3963_archive_v366_activate_v367.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "archive_v366_activate_v367_3963" in text


def test_req_report_3963_main_prints_written_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-3963: module main is a thin printable runner."""

    out_path = tmp_path / "results" / "experiment_3963_archive_v366_activate_v367.json"
    monkeypatch.setattr(mod, "run", lambda root: out_path)

    assert mod.main() == 0
    assert str(out_path) in capsys.readouterr().out
