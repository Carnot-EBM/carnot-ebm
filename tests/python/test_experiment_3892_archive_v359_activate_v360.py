"""Tests for Exp 3892 .359 archive and .360 green-gate activation.

Spec refs: REQ-REPORT-3892, SCENARIO-REPORT-3892,
SCENARIO-REPORT-3892-BLOCKED-YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v359_activate_v360_3892 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


SUMMARY_STDOUT = """\
==============================================================================
ARTIFACT  experiment_3882_thesis_a_partb_killgate.json
------------------------------------------------------------------------------
  verdict          : complete: thesis_a_partb_FUNDAMENTAL_beam0.000_argmin0.000_both_fail_vs_ar0.940_landscape_misshaped
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 3673.32   substrate: live_llm_inference
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3883_ebt_system2_kcurve.json
------------------------------------------------------------------------------
  verdict          : complete: ebt_system2_BOUNDED_PLATEAU_no_usable_descent_signal_at_scale
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 310.12   substrate: live_llm_inference
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3884_in_distribution_error_rich_corpus.json
------------------------------------------------------------------------------
  verdict          : complete: in_distribution_corpus_READY_nerr150_auroc0.9667_moat_scissor_can_run
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.66   substrate: cpu_carnot_verify_exp2837_cached_fover_rows
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3885_moat_scissor_in_distribution.json
------------------------------------------------------------------------------
  verdict          : complete: moat_scissor_indist_INCONCLUSIVE_reasoner_self_verify_auroc
  flagged_adversarial (stamped): True   |   LIVE re-check: CRITICAL
  duration_s       : 35.31   substrate: live verifier substrate omitted from artifact
  adversarial flags:
      [critical] DURATION_TOO_SHORT: prior flagged artifact
==============================================================================
ARTIFACT  experiment_3886_graph_grounding_fact_verifier_defabricated.json
------------------------------------------------------------------------------
  verdict          : blocked_graph_verifier_not_invoked
  flagged_adversarial (stamped): True   |   LIVE re-check: CRITICAL
  duration_s       : 11.48   substrate: live verifier substrate omitted from artifact
  adversarial flags:
      [critical] DURATION_TOO_SHORT: prior flagged artifact
==============================================================================
ARTIFACT  experiment_3887_facts_complementarity.json
------------------------------------------------------------------------------
  verdict          : blocked_upstream_scores_missing
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.000452   substrate: cached aggregation
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3888_fr11_v24_independence_reweighting.json
------------------------------------------------------------------------------
  verdict          : complete: fr11_v24_INVARIANT_HELD_auroc0.9075_memcontrib0.0185_state_persisted
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.77   substrate: cached verifier ensemble
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3889_gatemate_continuity_corrigendum.json
------------------------------------------------------------------------------
  verdict          : blocked_gatemate_board_unreachable
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.041887   substrate: hardware_smoke
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3890_polarfire_kv260_continuity.json
------------------------------------------------------------------------------
  verdict          : success: polarfire_kv260_continuity_pfterminal_hash_verified_soft_cpu_ssh_dispatch_kvnonterminal_carnot_ising_inactive_uio_present_no_fabric_claim
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 20.622822   substrate: hardware_smoke
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3891_capstone_v359.json
------------------------------------------------------------------------------
  verdict          : complete: capstone_v359_ebtFUNDAMENTAL_moatINCONCLUSIVE_factsEXCLUDED_EXP3886_FLAGGED_paper_ready_true_frozen_unchanged
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 0.451829   substrate: artifact aggregation only
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


def _seed_repo(root: Path, *, corrupt_complete: bool = False, milestone: str = "2026.06.360") -> None:
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{milestone}"\n'
        "tasks:\n"
        "  - id: exp3892-archive-v359-activate-v360-green-gate\n"
        "    agent_type: codex\n"
        "    requires_codex: true\n",
        encoding="utf-8",
    )
    complete_text = (
        "milestones:\n"
        "- id: 2026.06.359\n"
        "  title: stale conductor archive\n"
        "  completed: '2026-06-06'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3882-thesis-a-partb-killgate-import-fixed\n"
        "    deliverable: results/experiment_3882_thesis_a_partb_killgate.json\n"
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


def test_req_report_3892_spec_anchor_exists() -> None:
    """REQ-REPORT-3892: OpenSpec declares the archive and green-gate contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3892" in spec
    assert "SCENARIO-REPORT-3892" in spec
    assert "SCENARIO-REPORT-3892-BLOCKED-YAML" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "prior_milestone_verdicts_summary" in spec


def test_scenario_report_3892_run_appends_verdicts_and_green_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3892: archive .359 truth and assert .360 green gates."""

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
        ebt_import_result=_command_result(mod.ebt_import_command()),
        started_s=4.0,
        now_s=5.5,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    complete_text = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    complete = yaml.safe_load(complete_text)
    archived = complete["milestones"][-1]
    task_results = {task["id"]: task["result"] for task in archived["tasks"]}

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.TERMINAL_VERDICT
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert not isinstance(artifact[field], dict), field
    assert artifact["archived_milestone"] == "2026.06.359"
    assert artifact["activated_milestone"] == "2026.06.360"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["core_pretest_green"] is True
    assert artifact["ebt_harness_importable"] is True
    assert artifact["summary_exit_code"] == 2
    assert artifact["summary_critical_flags_archived"] is True
    assert artifact["exp3882_honest_verdict"].startswith("complete: thesis_a_partb_FUNDAMENTAL")
    assert artifact["exp3885_honest_verdict"].startswith("complete: moat_scissor")
    assert artifact["exp3886_honest_verdict"] == "blocked_graph_verifier_not_invoked"
    assert artifact["n_tasks_archived"] == 10
    assert artifact["duration_s"] == 1.5
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    for exp_id in range(3882, 3892):
        assert f"exp{exp_id}:" in artifact["prior_milestone_verdicts_summary"]

    assert complete_text.startswith(before["complete"].rstrip())
    assert complete_text.count("correction_type: v359_forward_bets_archive_activation") == 1
    assert complete_text.count("- id: 2026.06.359") == 2
    assert "result: complete:" not in complete_text
    assert "result: 'complete: thesis_a_partb_FUNDAMENTAL" in complete_text
    assert "result: 'blocked_graph_verifier_not_invoked'" in complete_text
    assert archived["activation_recorded"] == "exp3892-archive-v359-activate-v360-green-gate"
    assert task_results["exp3882-thesis-a-partb-killgate-import-fixed"].startswith("complete:")
    assert task_results["exp3886-graph-grounding-fact-verifier-defabricated"] == (
        "blocked_graph_verifier_not_invoked"
    )
    assert yaml.safe_load(complete_text)

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before["roadmap"]
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before["changelog"]
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before["status"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8") == before["trace"]
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before["conductor"]


def test_req_report_3892_append_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3892: rerunning does not append duplicate corrective records."""

    _seed_repo(tmp_path)

    first = mod.run(
        tmp_path,
        summary_result=_summary_result(),
        core_pretest_result=_command_result(mod.core_pretest_command()),
        ebt_import_result=_command_result(mod.ebt_import_command()),
        started_s=1.0,
        now_s=1.25,
    ).read_text(encoding="utf-8")
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second = mod.run(
        tmp_path,
        summary_result=_summary_result(),
        core_pretest_result=_command_result(mod.core_pretest_command()),
        ebt_import_result=_command_result(mod.ebt_import_command()),
        started_s=1.0,
        now_s=1.25,
    ).read_text(encoding="utf-8")
    second_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")

    assert first == second
    assert first_complete == second_complete
    assert second_complete.count("correction_type: v359_forward_bets_archive_activation") == 1


def test_scenario_report_3892_blocked_yaml_writes_artifact_without_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3892-BLOCKED-YAML: corrupt YAML exits before append."""

    _seed_repo(tmp_path, corrupt_complete=True)
    before = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")

    out_path = mod.run(
        tmp_path,
        summary_result=_summary_result(),
        core_pretest_result=_command_result(mod.core_pretest_command()),
        ebt_import_result=_command_result(mod.ebt_import_command()),
        started_s=7.0,
        now_s=7.1,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison")
    assert artifact["research_complete_yaml_parses"] is False
    assert artifact["core_pretest_green"] is False
    assert artifact["ebt_harness_importable"] is False
    assert artifact["preconditions_checked"]["research_complete_yaml_exists"] is True
    assert artifact["preconditions_checked"]["research_complete_yaml_parsed_before"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("prior_milestone_verdicts_summary"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(archived_milestone="2026.06.358"), "archived milestone"),
        (lambda p: p.update(activated_milestone="2026.06.359"), "activated milestone"),
        (lambda p: p.update(research_complete_yaml_parses=False), "YAML must parse"),
        (lambda p: p.update(core_pretest_green=False), "core pretest"),
        (lambda p: p.update(ebt_harness_importable=False), "EBT harness"),
        (lambda p: p.update(active_milestone_confirmed=False), "active milestone"),
        (lambda p: p.update(n_tasks_archived=9), "n_tasks_archived"),
        (lambda p: p.update(exp3882_honest_verdict="complete: wrong"), "Exp 3882"),
        (lambda p: p.update(exp3885_honest_verdict="blocked_wrong"), "Exp 3885"),
        (lambda p: p.update(exp3886_honest_verdict="complete: wrong"), "Exp 3886"),
        (lambda p: p.update(backend_routing_recommendation="gemini only"), "backend"),
        (lambda p: p.update(inference_substrate="live_model"), "inference"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(copied_marker="GGUF"), "compute-bound markers"),
    ],
)
def test_req_report_3892_validate_artifact_rejects_regressions(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3892: validation rejects fields that hide green-gate risk."""

    _seed_repo(tmp_path)
    payload = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            ebt_import_result=_command_result(mod.ebt_import_command()),
            started_s=9.0,
            now_s=9.5,
        ).read_text(encoding="utf-8")
    )

    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(broken)


def test_req_report_3892_helpers_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3892: helper failures block instead of fabricating success."""

    _seed_repo(tmp_path, milestone="2026.06.359")
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            ebt_import_result=_command_result(mod.ebt_import_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_v360_not_active")

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(stdout="no verdict here"),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            ebt_import_result=_command_result(mod.ebt_import_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_v359_summary_missing_verdict")

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            core_pretest_result=_command_result(mod.core_pretest_command(), exit_code=1, stdout="failed"),
            ebt_import_result=_command_result(mod.ebt_import_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_core_pretest_failed")
    assert artifact["core_pretest_green"] is False

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            ebt_import_result=_command_result(mod.ebt_import_command(), exit_code=1, stdout=""),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_ebt_harness_import")
    assert artifact["ebt_harness_importable"] is False

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
            ebt_import_result=_command_result(mod.ebt_import_command()),
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
            ebt_import_result=_command_result(mod.ebt_import_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison_missing")


def test_req_report_3892_subprocess_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3892: subprocess helpers use the mandated commands."""

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
    assert mod.run_ebt_import_check(tmp_path).stdout == SUMMARY_STDOUT
    assert calls == [mod.ebt_import_command()]

    def fail_subprocess(cmd: list[str], **kwargs: object) -> object:
        raise mod.subprocess.CalledProcessError(1, cmd, output="", stderr="failed")

    monkeypatch.setattr(mod.subprocess, "run", fail_subprocess)
    assert mod.run_core_pretest(tmp_path).exit_code == 1
    assert mod.run_ebt_import_check(tmp_path).exit_code == 1

    def os_error_subprocess(cmd: list[str], **kwargs: object) -> object:
        raise OSError("missing executable")

    monkeypatch.setattr(mod.subprocess, "run", os_error_subprocess)
    assert mod.run_core_pretest(tmp_path).exit_code == 127
    assert mod.run_ebt_import_check(tmp_path).exit_code == 127


def test_scenario_report_3892_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3892: the requested script entrypoint exists."""

    script = Path("scripts/experiments/experiment_3892_archive_v359_activate_v360.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "archive_v359_activate_v360_3892" in text
