"""Tests for Exp 3881 .358 archive and .359 green-gate activation.

Spec refs: REQ-REPORT-3881, SCENARIO-REPORT-3881,
SCENARIO-REPORT-3881-BLOCKED-YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v358_activate_v359_3881 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


SUMMARY_STDOUT = """\
==============================================================================
ARTIFACT  experiment_3870_archive_v357_activate_v358.json
------------------------------------------------------------------------------
  verdict          : complete: archived_v357_inconclusive_exp3869_positive_controls_degenerate_v358_active_codex_backend_recommended
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  acceptance gates : (none found - claim has no self-reported gate)
  duration_s       : 8.305831   substrate: aggregation_from_upstream_artifacts
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3871_thesis_a_partb_dtp1_headroom_confirmed.json
------------------------------------------------------------------------------
  verdict          : blocked_scaled_harness_import
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  acceptance gates : (none found - claim has no self-reported gate)
  duration_s       : {'principle': 'wrapped by failed task', 'value': 0.11}   substrate: {'principle': 'wrapped by failed task', 'value': 'blocked_precondition'}
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3872_ebt_energy_descent_system2_diagnostic.json
------------------------------------------------------------------------------
  verdict          : blocked_gate_check_failed
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  acceptance gates :
      [?   ] gate_check_summary = '1 of 1 gate(s) failed'
  duration_s       : 0.0   substrate: None
  adversarial flags: none
==============================================================================
"""


def _summary_result(exit_code: int = 0, stdout: str = SUMMARY_STDOUT) -> mod.CommandResult:
    return mod.CommandResult(
        command=[
            str(mod.PYTHON_BIN),
            "scripts/summarize_artifact.py",
            "3870",
            "3871",
            "3872",
        ],
        exit_code=exit_code,
        stdout=stdout,
        stderr="",
    )


def _command_result(command: list[str], *, exit_code: int = 0, stdout: str = "ok\n") -> mod.CommandResult:
    return mod.CommandResult(command=command, exit_code=exit_code, stdout=stdout, stderr="")


def _seed_repo(root: Path, *, corrupt_complete: bool = False, milestone: str = "2026.06.359") -> None:
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{milestone}"\n'
        "tasks:\n"
        "  - id: exp3881-archive-v358-activate-v359-green-gate\n"
        "    agent_type: codex\n"
        "    requires_codex: true\n",
        encoding="utf-8",
    )
    complete_text = (
        "milestones:\n"
        "- id: 2026.06.358\n"
        "  title: stale conductor archive\n"
        "  completed: '2026-06-06'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3871-thesis-a-partb-dtp1-adjudication-headroom-confirmed\n"
        "    deliverable: results/experiment_3871_thesis_a_partb_dtp1_headroom_confirmed.json\n"
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


def test_req_report_3881_spec_anchor_exists() -> None:
    """REQ-REPORT-3881: OpenSpec declares the archive and green-gate contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3881" in spec
    assert "SCENARIO-REPORT-3881" in spec
    assert "SCENARIO-REPORT-3881-BLOCKED-YAML" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "blocked_scaled_harness_import" in spec


def test_scenario_report_3881_run_appends_wipeout_and_green_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3881: archive .358 wipeout and assert all green gates."""

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
        summary_result=_summary_result(),
        core_pretest_result=_command_result(mod.core_pretest_command()),
        ebt_import_result=_command_result(mod.ebt_import_command()),
        started_s=4.0,
        now_s=5.25,
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
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["archived_milestone"] == "2026.06.358"
    assert artifact["activated_milestone"] == "2026.06.359"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["core_pretest_green"] is True
    assert artifact["ebt_harness_importable"] is True
    assert artifact["working_ebt_import_incantation"] == mod.WORKING_EBT_IMPORT_INCANTATION
    assert artifact["exp3871_honest_verdict"] == "blocked_scaled_harness_import"
    assert artifact["exp3872_honest_verdict"] == "blocked_gate_check_failed"
    assert artifact["missing_artifact_task_count"] == 8
    assert artifact["active_milestone_confirmed"] is True
    assert artifact["duration_s"] == 1.25
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    assert complete_text.startswith(before["complete"].rstrip())
    assert complete_text.count("correction_type: v358_execution_wipeout_archive_activation") == 1
    assert complete_text.count("- id: 2026.06.358") == 2
    assert "result: complete:" not in complete_text
    assert "result: 'complete: archived_v357_inconclusive" in complete_text
    assert "result: 'blocked_scaled_harness_import'" in complete_text
    assert "result: 'blocked_gate_check_failed'" in complete_text
    assert "SKIPPED_BY_PRETEST_GATE" in complete_text
    assert archived["activation_recorded"] == "exp3881-archive-v358-activate-v359-green-gate"
    assert task_results["exp3870-archive-v357-activate-v358-backend-diag"].startswith("complete:")
    assert task_results["exp3871-thesis-a-partb-dtp1-adjudication-headroom-confirmed"] == (
        "blocked_scaled_harness_import"
    )
    assert task_results["exp3880-capstone-v358"].startswith("SKIPPED_BY_PRETEST_GATE")
    assert yaml.safe_load(complete_text)

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before["roadmap"]
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before["changelog"]
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before["status"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8") == before["trace"]
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before["conductor"]


def test_req_report_3881_append_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3881: rerunning does not append duplicate corrective records."""

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
    assert second_complete.count("correction_type: v358_execution_wipeout_archive_activation") == 1


def test_scenario_report_3881_blocked_yaml_writes_artifact_without_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3881-BLOCKED-YAML: corrupt YAML exits before append."""

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
        (lambda p: p.pop("archived_milestone"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(archived_milestone="2026.06.357"), "archived milestone"),
        (lambda p: p.update(activated_milestone="2026.06.358"), "activated milestone"),
        (lambda p: p.update(research_complete_yaml_parses=False), "YAML must parse"),
        (lambda p: p.update(core_pretest_green=False), "core pretest"),
        (lambda p: p.update(ebt_harness_importable=False), "EBT harness"),
        (lambda p: p.update(active_milestone_confirmed=False), "active milestone"),
        (lambda p: p.update(exp3871_honest_verdict="complete: wrong"), "Exp 3871"),
        (lambda p: p.update(missing_artifact_task_count=7), "missing artifact"),
        (lambda p: p.update(backend_routing_recommendation="gemini only"), "backend"),
        (lambda p: p.update(inference_substrate="live_model"), "inference"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(copied_marker="GGUF"), "compute-bound markers"),
    ],
)
def test_req_report_3881_validate_artifact_rejects_regressions(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3881: validation rejects fields that hide green-gate risk."""

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


def test_req_report_3881_helpers_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3881: helper failures block instead of fabricating success."""

    _seed_repo(tmp_path, milestone="2026.06.358")
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            ebt_import_result=_command_result(mod.ebt_import_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_v359_not_active")

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(stdout="no verdict here"),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            ebt_import_result=_command_result(mod.ebt_import_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_v358_summary_missing_verdict")

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(exit_code=2),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            ebt_import_result=_command_result(mod.ebt_import_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_v358_summary_critical")

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


def test_req_report_3881_subprocess_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3881: subprocess helpers use the mandated commands."""

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
    assert summary.command == [
        str(mod.PYTHON_BIN),
        "scripts/summarize_artifact.py",
        "3870",
        "3871",
        "3872",
    ]

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


def test_scenario_report_3881_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3881: the requested script entrypoint exists."""

    script = Path("scripts/experiments/experiment_3881_archive_v358_activate_v359.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "archive_v358_activate_v359_3881" in text
