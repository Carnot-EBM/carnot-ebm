"""Tests for Exp 3934 .363 archive and .364 activation.

Spec refs: REQ-REPORT-3934, SCENARIO-REPORT-3934,
SCENARIO-REPORT-3934-BLOCKED-YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v363_activate_v364_3934 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


SUMMARY_STDOUT = """\
!! no artifact matched: results/experiment_3925_competent_judge_build.json
==============================================================================
ARTIFACT  experiment_3924_archive_v362_activate_v363_retire_facts.json
------------------------------------------------------------------------------
  verdict          : complete: archived_v362_v363_active_facts_retired_comparator_flaw_recorded_green_gates
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 13.387981   substrate: aggregation_from_upstream_artifacts
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3926_valid_efficiency_head_to_head.json
------------------------------------------------------------------------------
  verdict          : blocked_upstream_competent_judge_not_ready
  flagged_adversarial (stamped): True   |   LIVE re-check: CRITICAL
  duration_s       : 1.0734215900010895   substrate: none_blocked_preflight
  adversarial flags:
      [critical] DURATION_TOO_SHORT: duration_s=1.0734215900010895 but artifact references compute markers
==============================================================================
ARTIFACT  experiment_3927_non_degenerate_cascade_router.json
------------------------------------------------------------------------------
  verdict          : blocked_upstream_valid_efficiency_missing
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.0001163482666015625   substrate: none_blocked_preflight
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3928_moat_scissor_replication.json
------------------------------------------------------------------------------
  verdict          : blocked_all_gguf_inference_failed
  flagged_adversarial (stamped): True   |   LIVE re-check: CRITICAL
  duration_s       : 13.105805397033691   substrate: none_blocked_preflight
  adversarial flags:
      [critical] DURATION_TOO_SHORT: duration_s=13.105805397033691 but artifact references compute markers
==============================================================================
ARTIFACT  experiment_3929_arc_agi3_action_efficiency.json
------------------------------------------------------------------------------
  verdict          : complete: arc_agi3_verifier_router_HELPS_ratio1.959_ci1.742-2.194_synthetic_first_agentic_step_real_benchmark_reachabletrue
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 1.600117751979269   substrate: synthetic_arc_grid_cpu_energy_verifier
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3930_fr11_v26_cascade_band_online_learning.json
------------------------------------------------------------------------------
  verdict          : complete: fr11_v26_INVARIANT_HELD_auroc0.908_memcontrib0.0185_cascade_band_learnedtrue_state_persisted
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 1.170836449   substrate: cached_candidates
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3931_hardware_continuity_clean_rerun.json
------------------------------------------------------------------------------
  verdict          : success: hardware_continuity_clean_gatemateblocked_pfterminal_hash_verified_no_fabric_claim
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 20.880724   substrate: hardware_smoke
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3932_literature_synthesis_agentic_verification.json
------------------------------------------------------------------------------
  verdict          : complete: literature_synthesis_positioned_0_new_refs_public_docs_untouched
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.0001   substrate: no_new_inference_local_disk_synthesis_cpu
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3933_capstone_v363.json
------------------------------------------------------------------------------
  verdict          : complete: capstone_v363_efficiencyINCONCLUSIVE_moat_replicatedfalse_earnsfalse_paper_ready_true_frozen_unchanged
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.4747728710062802   substrate: aggregation_from_upstream_artifacts
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


def _summary_result(exit_code: int = 2, stdout: str = SUMMARY_STDOUT) -> mod.CommandResult:
    return _command_result(mod.summary_command(), exit_code=exit_code, stdout=stdout)


def _import_stdout(*, all_ok: bool = True) -> str:
    return json.dumps(
        {
            module: {"import_ok": all_ok, "error": None if all_ok else "ImportError"}
            for module in mod.EVAL_IMPORT_MODULES
        },
        sort_keys=True,
    )


def _import_result(*, all_ok: bool = True) -> mod.CommandResult:
    return _command_result(
        mod.eval_modules_import_command(),
        exit_code=0 if all_ok else 1,
        stdout=_import_stdout(all_ok=all_ok),
    )


def _seed_repo(
    root: Path,
    *,
    corrupt_complete: bool = False,
    milestone: str = "2026.06.364",
    manifest: str | None = None,
    max_token_fields: bool = True,
) -> None:
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{milestone}"\n'
        "tasks:\n"
        "  - id: exp3934-archive-v363-activate-v364\n"
        "    agent_type: codex\n",
        encoding="utf-8",
    )
    complete_text = (
        "milestones:\n"
        "- id: 2026.06.363\n"
        "  title: stale conductor archive\n"
        "  completed: '2026-06-08'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3924-archive-v362-activate-v363-retire-facts\n"
        "    deliverable: results/experiment_3924_archive_v362_activate_v363_retire_facts.json\n"
        "    result: OK (conductor)\n"
    )
    if corrupt_complete:
        complete_text += "  - id: poison\n    result: complete: unquoted colon\n"
    (root / "research-complete.yaml").write_text(complete_text, encoding="utf-8")
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "experiments").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "python" / "carnot" / "verify").mkdir(parents=True, exist_ok=True)
    (root / "python" / "carnot" / "eval").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "changelog.md").write_text("changelog before\n", encoding="utf-8")
    (root / "ops" / "status.md").write_text("status before\n", encoding="utf-8")
    (root / "_bmad" / "traceability.md").write_text("trace before\n", encoding="utf-8")
    (root / "scripts" / "research_conductor.py").write_text("# conductor before\n", encoding="utf-8")
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        manifest or "retired_experiments:\n  - experiment_id: 3920\n    reason: existing\n",
        encoding="utf-8",
    )
    (root / "python" / "carnot" / "verify" / "competent_llm_judge.py").write_text(
        "# drafted judge\n",
        encoding="utf-8",
    )
    (root / "scripts" / "experiments" / "experiment_3925_competent_judge_build.py").write_text(
        "# drafted runner\n",
        encoding="utf-8",
    )
    token_body = (
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\n"
        "class ExperimentConfig:\n"
    )
    if max_token_fields:
        token_body += "    max_tokens_weak: int = 96\n    max_tokens_strong: int = 160\n"
    else:
        token_body += "    panel_limit: int = 40\n"
    (root / "python" / "carnot" / "eval" / "moat_scissor_replication_3928.py").write_text(
        token_body,
        encoding="utf-8",
    )


def _run_success(root: Path, **overrides: object) -> Path:
    kwargs = {
        "research_complete_parse_result": _command_result(mod.research_complete_yaml_command()),
        "summary_result": _summary_result(),
        "core_pretest_result": _command_result(mod.core_pretest_command()),
        "eval_modules_import_result": _import_result(),
        "started_s": 1.0,
        "now_s": 2.25,
    }
    kwargs.update(overrides)
    return mod.run(root, **kwargs)


def test_req_report_3934_spec_anchor_exists() -> None:
    """REQ-REPORT-3934: OpenSpec declares the archive and green-gate contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3934" in spec
    assert "SCENARIO-REPORT-3934" in spec
    assert "SCENARIO-REPORT-3934-BLOCKED-YAML" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "max_tokens_weak" in spec


def test_scenario_report_3934_run_appends_archive_and_green_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3934: archive .363 and record the .364 unblock state."""

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
    assert artifact["archived_milestone"] == "2026.06.363"
    assert artifact["activated_milestone"] == "2026.06.364"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["core_pretest_green"] is True
    assert artifact["eval_modules_importable"] is True
    assert artifact["competent_judge_drafted_present"] is True
    assert artifact["max_tokens_weak_field_present"] is True
    assert artifact["max_tokens_strong_field_present"] is True
    assert artifact["n_tasks_archived"] == 10
    assert artifact["duration_s"] == 1.25
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert "exp3925: missing_artifact:" in artifact["prior_milestone_verdicts_summary"]
    assert "exp3928: blocked_all_gguf_inference_failed" in artifact[
        "prior_milestone_verdicts_summary"
    ]
    assert "exp3933: complete: capstone_v363_efficiencyINCONCLUSIVE" in artifact[
        "prior_milestone_verdicts_summary"
    ]
    assert "max_tokens_weak" in artifact["n363_blocker_state_recorded"]
    assert "exp3925 artifact missing" in artifact["n363_blocker_state_recorded"]
    assert artifact["eval_module_import_results"]["carnot.verify"]["import_ok"] is True

    assert complete_text.startswith(before["complete"].rstrip())
    assert complete_text.count(mod.ARCHIVE_MARKER) == 1
    assert complete_text.count("- id: 2026.06.363") == 2
    assert "result: complete:" not in complete_text
    assert "result: 'complete:" in complete_text
    assert archived["activation_recorded"] == "exp3934-archive-v363-activate-v364"
    assert task_results["exp3925-diagnose-and-build-competent-judge"].startswith(
        "missing_artifact:"
    )
    assert "LIVE_CRITICAL" in task_results["exp3926-valid-efficiency-head-to-head"]
    assert "LIVE_CRITICAL" in task_results["exp3928-moat-scissor-replication-second-corpus"]
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


def test_req_report_3934_append_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3934: rerunning does not duplicate the archive entry."""

    _seed_repo(tmp_path)

    first = _run_success(tmp_path).read_text(encoding="utf-8")
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second = _run_success(tmp_path).read_text(encoding="utf-8")

    assert first == second
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == first_complete
    assert first_complete.count(mod.ARCHIVE_MARKER) == 1


def test_scenario_report_3934_blocked_yaml_writes_artifact_without_edits(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3934-BLOCKED-YAML: corrupt YAML exits before edits."""

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
        core_pretest_result=_command_result(mod.core_pretest_command()),
        eval_modules_import_result=_import_result(),
        started_s=7.0,
        now_s=7.1,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison")
    assert artifact["research_complete_yaml_parses"] is False
    assert artifact["exclusion_manifest_parses"] is False
    assert artifact["core_pretest_green"] is False
    assert artifact["eval_modules_importable"] is False
    assert artifact["competent_judge_drafted_present"] is False
    assert artifact["max_tokens_weak_field_present"] is False
    assert artifact["preconditions_checked"]["research_complete_yaml_exists"] is True
    assert artifact["preconditions_checked"]["research_complete_yaml_parsed_before"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8") == before_manifest


def test_req_report_3934_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3934: hard helper failures block instead of fabricating success."""

    _seed_repo(tmp_path, milestone="2026.06.363")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_v364_not_active")

    _seed_repo(tmp_path, manifest="retired: [\n")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_exclusion_manifest_yaml_poison")

    _seed_repo(tmp_path)
    (tmp_path / "ops" / "exclusion_manifest.yaml").unlink()
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_exclusion_manifest_missing")

    _seed_repo(tmp_path)
    artifact = json.loads(
        _run_success(
            tmp_path,
            summary_result=_summary_result(exit_code=127, stdout=""),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_v363_summary_command_failed")

    _seed_repo(tmp_path)
    artifact = json.loads(
        _run_success(
            tmp_path,
            core_pretest_result=_command_result(mod.core_pretest_command(), exit_code=1),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_core_pretest_failed")
    assert artifact["core_pretest_green"] is False

    _seed_repo(tmp_path)
    artifact = json.loads(
        _run_success(tmp_path, eval_modules_import_result=_import_result(all_ok=False)).read_text(
            encoding="utf-8"
        )
    )
    assert artifact["honest_verdict"].startswith("blocked_eval_module_import")
    assert artifact["eval_modules_importable"] is False

    _seed_repo(tmp_path)
    (tmp_path / "scripts" / "experiments" / "experiment_3925_competent_judge_build.py").unlink()
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_competent_judge_draft_missing")

    _seed_repo(tmp_path, max_token_fields=False)
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_max_tokens_field_missing")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("n363_blocker_state_recorded"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p.update(archived_milestone="2026.06.362"), "archived milestone"),
        (lambda p: p.update(activated_milestone="2026.06.363"), "activated milestone"),
        (lambda p: p.update(research_complete_yaml_parses=False), "research-complete"),
        (lambda p: p.update(exclusion_manifest_parses=False), "manifest"),
        (lambda p: p.update(core_pretest_green=False), "core pretest"),
        (lambda p: p.update(eval_modules_importable=False), "module imports"),
        (lambda p: p.update(competent_judge_drafted_present=False), "competent judge"),
        (lambda p: p.update(max_tokens_weak_field_present=False), "max_tokens_weak"),
        (lambda p: p.update(active_milestone_confirmed=False), "active milestone"),
        (lambda p: p.update(n_tasks_archived=9), "n_tasks_archived"),
        (lambda p: p.update(prior_milestone_verdicts_summary="exp3924: ok"), "missing exp3925"),
        (lambda p: p.update(n363_blocker_state_recorded="all good"), "blocker state"),
        (lambda p: p.update(inference_substrate="live_model"), "inference"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(copied_marker="CUDA"), "compute-bound markers"),
    ],
)
def test_req_report_3934_validate_artifact_rejects_regressions(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3934: validation rejects fields that hide transition risk."""

    _seed_repo(tmp_path)
    payload = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))

    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(broken)


def test_req_report_3934_summary_parser_records_blockers_and_flags() -> None:
    """REQ-REPORT-3934: summaries preserve missing, blocked, and capstone evidence."""

    records = mod.parse_summary_records(SUMMARY_STDOUT)
    verdicts = mod.task_verdicts_from_summary(SUMMARY_STDOUT)
    summary = mod.build_prior_verdicts_summary(verdicts)

    assert records["3926"]["live_critical"] is True
    assert verdicts["exp3925-diagnose-and-build-competent-judge"].startswith("missing_artifact:")
    assert "LIVE_CRITICAL" in verdicts["exp3926-valid-efficiency-head-to-head"]
    assert "LIVE_CRITICAL" in verdicts["exp3928-moat-scissor-replication-second-corpus"]
    assert "exp3933: complete: capstone_v363_efficiencyINCONCLUSIVE" in summary
    assert mod.blocker_state_summary(verdicts).startswith("exp3925 artifact missing")


def test_req_report_3934_token_and_import_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-3934: file probes and import parsing stay deterministic."""

    _seed_repo(tmp_path)

    assert mod.drafted_competent_judge_present(tmp_path) is True
    fields = mod.experiment_config_token_fields(
        tmp_path / "python" / "carnot" / "eval" / "moat_scissor_replication_3928.py"
    )
    assert fields == {"max_tokens_weak": True, "max_tokens_strong": True}
    assert mod.experiment_config_token_fields(tmp_path / "missing.py") == {
        "max_tokens_weak": False,
        "max_tokens_strong": False,
    }
    assert mod.parse_eval_module_imports(_import_result())["carnot.verify"]["import_ok"] is True
    malformed = _command_result(mod.eval_modules_import_command(), stdout="{not json", exit_code=1)
    parsed = mod.parse_eval_module_imports(malformed)
    assert parsed["carnot.verify"]["import_ok"] is False
    assert "unparseable" in str(parsed["carnot.verify"]["error"])


def test_req_report_3934_edge_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3934: defensive fallback paths stay explicit and covered."""

    assert mod.duration_from(None, None) == 0.0001
    assert mod._milestone_from_text("tasks: []\n") == "unknown"
    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")

    odd_summary = """\
==============================================================================
ARTIFACT  experiment_3929_arc_agi3_action_efficiency.json
------------------------------------------------------------------------------
  verdict          : complete: arc_agi3_partial
  flagged_adversarial (stamped): True   |   LIVE re-check: warn
  duration_s       : not-a-number   substrate: aggregation
  headline metrics :
      solve_rate = 0.5
      bad_metric = not-a-number
  adversarial flags: none
==============================================================================
"""
    records = mod.parse_summary_records(odd_summary)
    verdicts = mod.task_verdicts_from_summary(odd_summary)
    assert records["3929"]["duration_s"] is None
    assert records["3929"]["headline_metrics"]["solve_rate"] == pytest.approx(0.5)
    assert "stamped_flagged" in verdicts["exp3929-arc-agi3-verifier-router-action-efficiency"]

    syntax_bad = tmp_path / "syntax_bad.py"
    syntax_bad.write_text("class ExperimentConfig(:\n", encoding="utf-8")
    assert mod.experiment_config_token_fields(syntax_bad) == {
        "max_tokens_weak": False,
        "max_tokens_strong": False,
    }
    assign_style = tmp_path / "assign_style.py"
    assign_style.write_text(
        "class ExperimentConfig:\n"
        "    max_tokens_weak = 96\n"
        "    max_tokens_strong = 160\n",
        encoding="utf-8",
    )
    assert mod.experiment_config_token_fields(assign_style) == {
        "max_tokens_weak": True,
        "max_tokens_strong": True,
    }
    partial_imports = _command_result(
        mod.eval_modules_import_command(),
        stdout=json.dumps({"carnot.verify": {"import_ok": True, "error": None}}),
    )
    assert mod.parse_eval_module_imports(partial_imports)[
        "carnot.verify.gguf_inference"
    ]["import_ok"] is False

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").unlink()
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison_missing")

    _seed_repo(tmp_path)
    monkeypatch.setattr(mod, "append_research_complete_record", lambda *args: "milestones: [\n")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_append_invalid")


def test_req_report_3934_subprocess_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3934: subprocess helpers use the mandated commands."""

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
    assert mod.run_core_pretest(tmp_path).exit_code == 0
    assert calls == [mod.core_pretest_command()]

    calls.clear()
    assert mod.run_eval_modules_import_check(tmp_path).stdout == _import_stdout()
    assert calls == [mod.eval_modules_import_command()]

    def fail_subprocess(cmd: list[str], **kwargs: object) -> object:
        raise mod.subprocess.CalledProcessError(1, cmd, output="", stderr="failed")

    monkeypatch.setattr(mod.subprocess, "run", fail_subprocess)
    assert mod.run_core_pretest(tmp_path).exit_code == 1
    assert mod.run_eval_modules_import_check(tmp_path).exit_code == 1

    def os_error_subprocess(cmd: list[str], **kwargs: object) -> object:
        raise OSError("missing executable")

    monkeypatch.setattr(mod.subprocess, "run", os_error_subprocess)
    assert mod.run_research_complete_parse_check(tmp_path).exit_code == 127
    assert mod.run_summarize_artifacts(tmp_path).exit_code == 127


def test_scenario_report_3934_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3934: the requested script entrypoint exists."""

    script = Path("scripts/experiments/experiment_3934_archive_v363_activate_v364.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "archive_v363_activate_v364_3934" in text


def test_req_report_3934_main_prints_written_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-3934: module main is a thin printable runner."""

    out_path = tmp_path / "results" / "experiment_3934_archive_v363_activate_v364.json"
    monkeypatch.setattr(mod, "run", lambda root: out_path)

    assert mod.main() == 0
    assert str(out_path) in capsys.readouterr().out
