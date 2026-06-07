"""Tests for Exp 3903 .360 archive and .361 green-gate activation.

Spec refs: REQ-REPORT-3903, SCENARIO-REPORT-3903,
SCENARIO-REPORT-3903-BLOCKED-YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v360_activate_v361_3903 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


SUMMARY_STDOUT = """\
==============================================================================
ARTIFACT  experiment_3892_archive_v359_activate_v360.json
------------------------------------------------------------------------------
  verdict          : complete: archived_v359_forward_bets_v360_active_green_gates_asserted_codex_backend_recommended
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 11.52   substrate: aggregation_from_upstream_artifacts
  adversarial flags: none
==============================================================================
!! no artifact matched: results/experiment_3893_ebt_fundamental_replication.json
==============================================================================
ARTIFACT  experiment_3894_reasoner_self_verify_harness.json
------------------------------------------------------------------------------
  verdict          : complete: reasoner_self_verify_harness_READY_fixture_auroc0.9167_ncaught6_moat_scissor_can_run
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 72.48   substrate: live_llm_inference
  headline metrics :
      fixture_auroc = 0.9166666666666666
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3895_moat_scissor_tested_harness.json
------------------------------------------------------------------------------
  verdict          : complete: moat_scissor_INCONCLUSIVE_reasoner_self_verify_auroc
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 135.97   substrate: live_llama_cpp_qwen3.6_35b_tested_reasoner_self_verification_plus_exp3884_disk_carnot_scores
  headline metrics :
      carnot_ensemble_auroc = 0.9666888888888889
      error_overlap_jaccard = 0.15942028985507245
      reasoner_self_verify_auroc = 0.5463111111111111
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3896_graph_grounding_verifier_harness.json
------------------------------------------------------------------------------
  verdict          : complete: graph_grounding_verifier_NOT_READY_fixture_auroc1.0000_model_invokedtrue
  flagged_adversarial (stamped): True   |   LIVE re-check: CRITICAL
  duration_s       : 43.82   substrate: live_llama_cpp_sota_gguf_graph_grounding_fixture
  adversarial flags:
      [critical] DURATION_TOO_SHORT: duration_s=43.82 but artifact references compute-bound markers
==============================================================================
!! no artifact matched: results/experiment_3897_graph_grounding_facts_run.json
!! no artifact matched: results/experiment_3898_facts_complementarity.json
!! no artifact matched: results/experiment_3899_fr11_v25.json
==============================================================================
ARTIFACT  experiment_3900_gatemate_terminal_confirmation.json
------------------------------------------------------------------------------
  verdict          : blocked_gatemate_board_unreachable
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.08   substrate: hardware_smoke
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3901_polarfire_kv260_continuity.json
------------------------------------------------------------------------------
  verdict          : success: polarfire_kv260_continuity_pfterminal_hash_verified_soft_cpu_ssh_dispatch_kvnonterminal_carnot_ising_inactive_uio_present_no_fabric_claim
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 20.85   substrate: hardware_smoke
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3902_capstone_v360.json
------------------------------------------------------------------------------
  verdict          : complete: capstone_v360_ebtINCONCLUSIVE_moatINCONCLUSIVE_factsINCONCLUSIVE_paper_ready_true_frozen_unchanged
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.44   substrate: aggregation_from_upstream_artifacts
  adversarial flags: none
==============================================================================
"""


def _summary_result(exit_code: int = 2, stdout: str = SUMMARY_STDOUT) -> mod.CommandResult:
    return mod.CommandResult(
        command=mod.summary_command(),
        exit_code=exit_code,
        stdout=stdout,
        stderr="",
    )


def _command_result(
    command: list[str],
    *,
    exit_code: int = 0,
    stdout: str = "ok\n",
) -> mod.CommandResult:
    return mod.CommandResult(command=command, exit_code=exit_code, stdout=stdout, stderr="")


def _seed_repo(root: Path, *, corrupt_complete: bool = False, milestone: str = "2026.06.361") -> None:
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{milestone}"\n'
        "tasks:\n"
        "  - id: exp3903-archive-v360-activate-v361-green-gate\n"
        "    agent_type: codex\n"
        "    requires_codex: true\n",
        encoding="utf-8",
    )
    complete_text = (
        "milestones:\n"
        "- id: 2026.06.360\n"
        "  title: stale conductor archive\n"
        "  completed: '2026-06-07'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3892-archive-v359-activate-v360-green-gate\n"
        "    deliverable: results/experiment_3892_archive_v359_activate_v360.json\n"
        "    result: OK (conductor)\n"
    )
    if corrupt_complete:
        complete_text += "  - id: poison\n    result: complete: unquoted colon\n"
    (root / "research-complete.yaml").write_text(complete_text, encoding="utf-8")
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "changelog.md").write_text("changelog before\n", encoding="utf-8")
    (root / "ops" / "status.md").write_text("status before\n", encoding="utf-8")
    (root / "_bmad" / "traceability.md").write_text("trace before\n", encoding="utf-8")
    (root / "scripts" / "research_conductor.py").write_text("# conductor before\n", encoding="utf-8")


def test_req_report_3903_spec_anchor_exists() -> None:
    """REQ-REPORT-3903: OpenSpec declares the archive and green-gate contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3903" in spec
    assert "SCENARIO-REPORT-3903" in spec
    assert "SCENARIO-REPORT-3903-BLOCKED-YAML" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "reasoner_harness_importable" in spec
    assert "MOAT_SURVIVES numbers" in spec


def test_scenario_report_3903_run_appends_verdicts_and_green_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3903: archive .360 truth and assert .361 green gates."""

    _seed_repo(tmp_path)
    before = {
        "complete": (tmp_path / "research-complete.yaml").read_text(encoding="utf-8"),
        "roadmap": (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8"),
        "changelog": (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8"),
        "status": (tmp_path / "ops" / "status.md").read_text(encoding="utf-8"),
        "trace": (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8"),
        "conductor": (tmp_path / "scripts" / "research_conductor.py").read_text(
            encoding="utf-8"
        ),
    }

    out_path = mod.run(
        tmp_path,
        summary_result=_summary_result(exit_code=2),
        core_pretest_result=_command_result(mod.core_pretest_command()),
        reasoner_import_result=_command_result(mod.reasoner_import_command()),
        started_s=4.0,
        now_s=5.5,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    complete_text = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    complete = yaml.safe_load(complete_text)
    archived = complete["milestones"][-1]
    task_results = {task["id"]: task["result"] for task in archived["tasks"]}

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.terminal_verdict(reasoner_importable=True)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert not isinstance(artifact[field], dict), field
    assert artifact["archived_milestone"] == "2026.06.360"
    assert artifact["activated_milestone"] == "2026.06.361"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["core_pretest_green"] is True
    assert artifact["reasoner_harness_importable"] is True
    assert artifact["summary_exit_code"] == 2
    assert artifact["summary_critical_flags_archived"] is True
    assert artifact["exp3893_honest_verdict"].startswith("missing_artifact:")
    assert artifact["exp3895_honest_verdict"].startswith("complete: moat_scissor")
    assert "LIVE_CRITICAL" in artifact["exp3896_honest_verdict"]
    assert artifact["n_tasks_archived"] == 11
    assert artifact["duration_s"] == 1.5
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    for exp_id in range(3892, 3903):
        assert f"exp{exp_id}:" in artifact["prior_milestone_verdicts_summary"]
    assert "mis-gated MOAT_SURVIVES numbers" in artifact["prior_milestone_verdicts_summary"]
    assert "residual_catch=0.905" in artifact["prior_milestone_verdicts_summary"]
    assert "EBT replication did not finish" in artifact["prior_milestone_verdicts_summary"]
    assert "facts fabricated again" in artifact["prior_milestone_verdicts_summary"]

    assert complete_text.startswith(before["complete"].rstrip())
    assert complete_text.count("correction_type: v360_harness_first_archive_activation") == 1
    assert complete_text.count("- id: 2026.06.360") == 2
    assert "result: complete:" not in complete_text
    assert "result: 'complete: reasoner_self_verify_harness_READY" in complete_text
    assert "result: 'missing_artifact:" in complete_text
    assert archived["activation_recorded"] == "exp3903-archive-v360-activate-v361-green-gate"
    assert task_results["exp3893-ebt-fundamental-adversarial-replication"].startswith(
        "missing_artifact:"
    )
    assert "MOAT_SURVIVES numbers" in task_results["exp3895-moat-scissor-in-distribution-tested-harness"]
    assert "LIVE_CRITICAL" in task_results["exp3896-build-test-graph-grounding-verifier"]
    assert yaml.safe_load(complete_text)

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before["roadmap"]
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before["changelog"]
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before["status"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8") == before["trace"]
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before["conductor"]


def test_req_report_3903_append_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3903: rerunning does not append duplicate corrective records."""

    _seed_repo(tmp_path)

    first = mod.run(
        tmp_path,
        summary_result=_summary_result(),
        core_pretest_result=_command_result(mod.core_pretest_command()),
        reasoner_import_result=_command_result(mod.reasoner_import_command()),
        started_s=1.0,
        now_s=1.25,
    ).read_text(encoding="utf-8")
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second = mod.run(
        tmp_path,
        summary_result=_summary_result(),
        core_pretest_result=_command_result(mod.core_pretest_command()),
        reasoner_import_result=_command_result(mod.reasoner_import_command()),
        started_s=1.0,
        now_s=1.25,
    ).read_text(encoding="utf-8")
    second_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")

    assert first == second
    assert first_complete == second_complete
    assert second_complete.count("correction_type: v360_harness_first_archive_activation") == 1


def test_scenario_report_3903_blocked_yaml_writes_artifact_without_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3903-BLOCKED-YAML: corrupt YAML exits before append."""

    _seed_repo(tmp_path, corrupt_complete=True)
    before = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")

    out_path = mod.run(
        tmp_path,
        summary_result=_summary_result(),
        core_pretest_result=_command_result(mod.core_pretest_command()),
        reasoner_import_result=_command_result(mod.reasoner_import_command()),
        started_s=7.0,
        now_s=7.1,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison")
    assert artifact["research_complete_yaml_parses"] is False
    assert artifact["core_pretest_green"] is False
    assert artifact["reasoner_harness_importable"] is False
    assert artifact["preconditions_checked"]["research_complete_yaml_exists"] is True
    assert artifact["preconditions_checked"]["research_complete_yaml_parsed_before"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before


def test_req_report_3903_reasoner_import_failure_is_recorded_not_fatal(tmp_path: Path) -> None:
    """REQ-REPORT-3903: reasoner import diagnostics are bare bools, not a hard gate."""

    _seed_repo(tmp_path)

    out_path = mod.run(
        tmp_path,
        summary_result=_summary_result(),
        core_pretest_result=_command_result(mod.core_pretest_command()),
        reasoner_import_result=_command_result(
            mod.reasoner_import_command(),
            exit_code=1,
            stdout="ImportError\n",
        ),
        started_s=8.0,
        now_s=9.0,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.terminal_verdict(reasoner_importable=False)
    assert artifact["reasoner_harness_importable"] is False
    assert artifact["reasoner_import_exit_code"] == 1
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["core_pretest_green"] is True


def test_req_report_3903_summary_parser_records_stamped_only_flags() -> None:
    """REQ-REPORT-3903: stamped historical flags remain visible in archive text."""

    stdout = """\
==============================================================================
ARTIFACT  experiment_3892_archive_v359_activate_v360.json
------------------------------------------------------------------------------
  verdict          : complete: historical_flagged_clean_now
  flagged_adversarial (stamped): True   |   LIVE re-check: clean
==============================================================================
"""

    verdicts = mod.task_verdicts_from_summary(stdout)

    assert (
        "summarize_artifact stamped_flagged"
        in verdicts["exp3892-archive-v359-activate-v360-green-gate"]
    )


def test_req_report_3903_helpers_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3903: hard helper failures block instead of fabricating success."""

    _seed_repo(tmp_path, milestone="2026.06.360")
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            reasoner_import_result=_command_result(mod.reasoner_import_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_v361_not_active")

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(exit_code=127, stdout=""),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            reasoner_import_result=_command_result(mod.reasoner_import_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_v360_summary_command_failed")

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            core_pretest_result=_command_result(mod.core_pretest_command(), exit_code=1, stdout="failed"),
            reasoner_import_result=_command_result(mod.reasoner_import_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_core_pretest_failed")
    assert artifact["core_pretest_green"] is False

    _seed_repo(tmp_path)
    before = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    monkeypatch.setattr(
        mod,
        "append_research_complete_record",
        lambda _text, _verdicts: "milestones:\n- id: bad\n  result: complete: broken\n",
    )
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            reasoner_import_result=_command_result(mod.reasoner_import_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_research_complete_append_invalid")
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before

    missing = tmp_path / "missing"
    missing.mkdir()
    artifact = json.loads(
        mod.run(
            missing,
            summary_result=_summary_result(),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            reasoner_import_result=_command_result(mod.reasoner_import_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison_missing")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("prior_milestone_verdicts_summary"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p.update(archived_milestone="2026.06.359"), "archived milestone"),
        (lambda p: p.update(activated_milestone="2026.06.360"), "activated milestone"),
        (lambda p: p.update(research_complete_yaml_parses=False), "YAML must parse"),
        (lambda p: p.update(core_pretest_green=False), "core pretest"),
        (lambda p: p.update(reasoner_harness_importable="yes"), "reasoner harness"),
        (lambda p: p.update(active_milestone_confirmed=False), "active milestone"),
        (lambda p: p.update(n_tasks_archived=10), "n_tasks_archived"),
        (lambda p: p.update(exp3895_honest_verdict="complete: wrong"), "Exp 3895"),
        (lambda p: p.update(exp3896_honest_verdict="complete: no flag"), "Exp 3896"),
        (lambda p: p.update(backend_routing_recommendation="gemini only"), "backend"),
        (lambda p: p.update(inference_substrate="live_model"), "inference"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(copied_marker="GGUF"), "compute-bound markers"),
    ],
)
def test_req_report_3903_validate_artifact_rejects_regressions(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3903: validation rejects fields that hide green-gate risk."""

    _seed_repo(tmp_path)
    payload = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            reasoner_import_result=_command_result(mod.reasoner_import_command()),
            started_s=9.0,
            now_s=9.5,
        ).read_text(encoding="utf-8")
    )

    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(broken)


def test_req_report_3903_subprocess_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3903: subprocess helpers use the mandated commands."""

    calls: list[list[str]] = []

    class Completed:
        returncode = 0
        stdout = SUMMARY_STDOUT
        stderr = ""

    def run_subprocess(cmd: list[str], **kwargs: object) -> Completed:
        calls.append(cmd)
        assert kwargs["cwd"] == tmp_path
        return Completed()

    monkeypatch.setattr(mod.subprocess, "run", run_subprocess)
    summary = mod.run_summarize_artifacts(tmp_path)
    assert summary.stdout == SUMMARY_STDOUT
    assert summary.exit_code == 0
    assert summary.command == mod.summary_command()

    calls.clear()
    assert mod.run_core_pretest(tmp_path).exit_code == 0
    assert calls == [mod.core_pretest_command()]

    calls.clear()
    assert mod.run_reasoner_import_check(tmp_path).stdout == SUMMARY_STDOUT
    assert calls == [mod.reasoner_import_command()]

    def fail_subprocess(cmd: list[str], **kwargs: object) -> object:
        raise mod.subprocess.CalledProcessError(1, cmd, output="", stderr="failed")

    monkeypatch.setattr(mod.subprocess, "run", fail_subprocess)
    assert mod.run_core_pretest(tmp_path).exit_code == 1
    assert mod.run_reasoner_import_check(tmp_path).exit_code == 1

    def os_error_subprocess(cmd: list[str], **kwargs: object) -> object:
        raise OSError("missing executable")

    monkeypatch.setattr(mod.subprocess, "run", os_error_subprocess)
    assert mod.run_core_pretest(tmp_path).exit_code == 127
    assert mod.run_reasoner_import_check(tmp_path).exit_code == 127


def test_scenario_report_3903_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3903: the requested script entrypoint exists."""

    script = Path("scripts/experiments/experiment_3903_archive_v360_activate_v361.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "archive_v360_activate_v361_3903" in text


def test_req_report_3903_main_prints_written_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-3903: module main is a thin printable runner."""

    out_path = tmp_path / "results" / "experiment_3903_archive_v360_activate_v361.json"
    monkeypatch.setattr(mod, "run", lambda root: out_path)

    assert mod.main() == 0
    assert str(out_path) in capsys.readouterr().out
