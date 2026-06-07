"""Tests for Exp 3914 .361 archive, .362 activation, and poison-test quarantine.

Spec refs: REQ-REPORT-3914, SCENARIO-REPORT-3914,
SCENARIO-REPORT-3914-BLOCKED-YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v361_activate_v362_3914 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
COST_TEST_PATH = Path("tests/python/test_cost_instrumented_verification.py")


SUMMARY_STDOUT = """\
==============================================================================
ARTIFACT  experiment_3903_archive_v360_activate_v361.json
------------------------------------------------------------------------------
  verdict          : complete: archived_v360_harness_first_v361_active_green_gates_asserted_reasoner_import_ok_codex_backend_recommended
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 12.472321   substrate: aggregation_from_upstream_artifacts
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3904_moat_scissor_regated.json
------------------------------------------------------------------------------
  verdict          : blocked_llama_cpp_inference_failed
  flagged_adversarial (stamped): None   |   LIVE re-check: warn
  duration_s       : 252.93368577957153   substrate: none_blocked_preflight
  adversarial flags:
      [warn    ] METHODOLOGY_MISSING: Compute-bound artifact missing: random_seed.
==============================================================================
ARTIFACT  experiment_3905_cost_instrumented_verify_harness.json
------------------------------------------------------------------------------
  verdict          : complete: cost_harness_NOT_READY_ratio30881.00_unit_testFalse
  flagged_adversarial (stamped): True   |   LIVE re-check: CRITICAL
  duration_s       : 35.861052296997514   substrate: live_llama_cpp_judge_cpu_forward_plus_cpu_verifier_ensemble
  headline metrics :
      fixture_energy_per_item_ms = 0.09216709877364337
  adversarial flags:
      [critical] DURATION_TOO_SHORT: duration_s=35.861052296997514 but artifact references compute-bound markers
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


def _seed_repo(root: Path, *, corrupt_complete: bool = False, milestone: str = "2026.06.362") -> None:
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{milestone}"\n'
        "tasks:\n"
        "  - id: exp3914-archive-v361-activate-v362-quarantine-poison-test\n"
        "    agent_type: codex\n"
        "    requires_codex: true\n",
        encoding="utf-8",
    )
    complete_text = (
        "milestones:\n"
        "- id: 2026.06.361\n"
        "  title: stale conductor archive\n"
        "  completed: '2026-06-07'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3903-archive-v360-activate-v361-green-gate\n"
        "    deliverable: results/experiment_3903_archive_v360_activate_v361.json\n"
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


def test_req_report_3914_spec_anchor_exists() -> None:
    """REQ-REPORT-3914: OpenSpec declares the archive and quarantine contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3914" in spec
    assert "SCENARIO-REPORT-3914" in spec
    assert "SCENARIO-REPORT-3914-BLOCKED-YAML" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "poison_test_quarantined" in spec
    assert "blocked_llama_cpp_inference_failed" in spec


def test_req_report_3914_cost_fixture_test_has_no_duration_floor() -> None:
    """REQ-REPORT-3914: the 10-row fixture test no longer asserts the 60s floor."""

    text = COST_TEST_PATH.read_text(encoding="utf-8")

    assert mod.poison_duration_assertion_present(text) is False
    assert 'assert artifact["duration_s"] >= 60' not in text
    assert 'assert artifact["fixture_cost_ratio"] > 1' in text
    assert 'assert artifact["fixture_llm_per_item_ms"] != artifact["fixture_energy_per_item_ms"]' in text


def test_scenario_report_3914_run_appends_wash_record_and_green_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3914: archive .361 truth, quarantine poison test, and assert gates."""

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
        poison_test_result=_command_result(mod.poison_test_command()),
        core_pretest_result=_command_result(mod.core_pretest_command()),
        live_model_import_result=_command_result(mod.live_model_import_command()),
        poison_test_text='assert artifact["fixture_cost_ratio"] > 1\n',
        started_s=4.0,
        now_s=5.5,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    complete_text = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    complete = yaml.safe_load(complete_text)
    archived = complete["milestones"][-1]
    task_results = {task["id"]: task["result"] for task in archived["tasks"]}

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.terminal_verdict(live_model_modules_importable=True)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert not isinstance(artifact[field], dict), field
    assert artifact["archived_milestone"] == "2026.06.361"
    assert artifact["activated_milestone"] == "2026.06.362"
    assert artifact["poison_test_quarantined"] is True
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["core_pretest_green"] is True
    assert artifact["live_model_modules_importable"] is True
    assert artifact["summary_exit_code"] == 2
    assert artifact["summary_critical_flags_archived"] is True
    assert artifact["exp3904_honest_verdict"] == "blocked_llama_cpp_inference_failed"
    assert "DURATION_TOO_SHORT" in artifact["exp3905_honest_verdict"]
    assert artifact["n_tasks_archived"] == 11
    assert artifact["duration_s"] == 1.5
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert "poison-test cascade" in artifact["n361_wash_root_causes"]
    assert "blocked_llama_cpp_inference_failed" in artifact["n361_wash_root_causes"]
    for exp_id in range(3903, 3914):
        assert f"exp{exp_id}:" in artifact["prior_milestone_verdicts_summary"]
    assert "1 failed, 105 passed" in artifact["prior_milestone_verdicts_summary"]

    assert complete_text.startswith(before["complete"].rstrip())
    assert complete_text.count("correction_type: v361_poison_test_cascade_archive_activation") == 1
    assert complete_text.count("- id: 2026.06.361") == 2
    assert "result: complete:" not in complete_text
    assert "result: 'complete: archived_v360" in complete_text
    assert "result: 'blocked_llama_cpp_inference_failed" in complete_text
    assert "result: 'SKIP: poison-test cascade" in complete_text
    assert archived["activation_recorded"] == "exp3914-archive-v361-activate-v362-quarantine-poison-test"
    assert task_results["exp3904-moat-scissor-regated-accuracy-axis"] == "blocked_llama_cpp_inference_failed"
    assert "DURATION_TOO_SHORT" in task_results["exp3905-build-test-cost-instrumented-verify-harness"]
    assert "poison-test cascade" in task_results["exp3906-efficiency-head-to-head"]
    assert yaml.safe_load(complete_text)

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before["roadmap"]
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before["changelog"]
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before["status"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8") == before["trace"]
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before["conductor"]


def test_req_report_3914_append_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3914: rerunning does not append duplicate corrective records."""

    _seed_repo(tmp_path)

    first = mod.run(
        tmp_path,
        summary_result=_summary_result(),
        poison_test_result=_command_result(mod.poison_test_command()),
        core_pretest_result=_command_result(mod.core_pretest_command()),
        live_model_import_result=_command_result(mod.live_model_import_command()),
        poison_test_text='assert artifact["fixture_cost_ratio"] > 1\n',
        started_s=1.0,
        now_s=1.25,
    ).read_text(encoding="utf-8")
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second = mod.run(
        tmp_path,
        summary_result=_summary_result(),
        poison_test_result=_command_result(mod.poison_test_command()),
        core_pretest_result=_command_result(mod.core_pretest_command()),
        live_model_import_result=_command_result(mod.live_model_import_command()),
        poison_test_text='assert artifact["fixture_cost_ratio"] > 1\n',
        started_s=1.0,
        now_s=1.25,
    ).read_text(encoding="utf-8")
    second_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")

    assert first == second
    assert first_complete == second_complete
    assert second_complete.count("correction_type: v361_poison_test_cascade_archive_activation") == 1


def test_scenario_report_3914_blocked_yaml_writes_artifact_without_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3914-BLOCKED-YAML: corrupt YAML exits before append."""

    _seed_repo(tmp_path, corrupt_complete=True)
    before = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")

    out_path = mod.run(
        tmp_path,
        summary_result=_summary_result(),
        poison_test_result=_command_result(mod.poison_test_command()),
        core_pretest_result=_command_result(mod.core_pretest_command()),
        live_model_import_result=_command_result(mod.live_model_import_command()),
        poison_test_text='assert artifact["fixture_cost_ratio"] > 1\n',
        started_s=7.0,
        now_s=7.1,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison")
    assert artifact["research_complete_yaml_parses"] is False
    assert artifact["poison_test_quarantined"] is False
    assert artifact["core_pretest_green"] is False
    assert artifact["live_model_modules_importable"] is False
    assert artifact["preconditions_checked"]["research_complete_yaml_exists"] is True
    assert artifact["preconditions_checked"]["research_complete_yaml_parsed_before"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("n361_wash_root_causes"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p.update(archived_milestone="2026.06.360"), "archived milestone"),
        (lambda p: p.update(activated_milestone="2026.06.361"), "activated milestone"),
        (lambda p: p.update(poison_test_quarantined=False), "poison test"),
        (lambda p: p.update(research_complete_yaml_parses=False), "YAML must parse"),
        (lambda p: p.update(core_pretest_green=False), "core pretest"),
        (lambda p: p.update(live_model_modules_importable="yes"), "live model modules"),
        (lambda p: p.update(active_milestone_confirmed=False), "active milestone"),
        (lambda p: p.update(n_tasks_archived=10), "n_tasks_archived"),
        (lambda p: p.update(exp3904_honest_verdict="complete: wrong"), "Exp 3904"),
        (lambda p: p.update(exp3905_honest_verdict="complete: no flag"), "Exp 3905"),
        (lambda p: p.update(n361_wash_root_causes="only poison-test cascade"), "root causes"),
        (lambda p: p.update(backend_routing_recommendation="gemini only"), "backend"),
        (lambda p: p.update(inference_substrate="live_model"), "inference"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(copied_marker="GGUF"), "compute-bound markers"),
    ],
)
def test_req_report_3914_validate_artifact_rejects_regressions(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3914: validation rejects fields that hide cascade risk."""

    _seed_repo(tmp_path)
    payload = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            poison_test_result=_command_result(mod.poison_test_command()),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            live_model_import_result=_command_result(mod.live_model_import_command()),
            poison_test_text='assert artifact["fixture_cost_ratio"] > 1\n',
            started_s=9.0,
            now_s=9.5,
        ).read_text(encoding="utf-8")
    )

    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(broken)


def test_req_report_3914_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3914: hard helper failures block instead of fabricating success."""

    _seed_repo(tmp_path, milestone="2026.06.361")
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            poison_test_result=_command_result(mod.poison_test_command()),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            live_model_import_result=_command_result(mod.live_model_import_command()),
            poison_test_text='assert artifact["fixture_cost_ratio"] > 1\n',
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_v362_not_active")

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            poison_test_result=_command_result(mod.poison_test_command(), exit_code=1, stdout="failed"),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            live_model_import_result=_command_result(mod.live_model_import_command()),
            poison_test_text='assert artifact["fixture_cost_ratio"] > 1\n',
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_poison_test_quarantine_failed")
    assert artifact["poison_test_quarantined"] is False

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            poison_test_result=_command_result(mod.poison_test_command()),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            live_model_import_result=_command_result(mod.live_model_import_command()),
            poison_test_text='assert artifact["duration_s"] >= 60\n',
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_poison_test_assertion_present")

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(exit_code=127, stdout=""),
            poison_test_result=_command_result(mod.poison_test_command()),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            live_model_import_result=_command_result(mod.live_model_import_command()),
            poison_test_text='assert artifact["fixture_cost_ratio"] > 1\n',
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_v361_summary_command_failed")

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            poison_test_result=_command_result(mod.poison_test_command()),
            core_pretest_result=_command_result(mod.core_pretest_command(), exit_code=1, stdout="failed"),
            live_model_import_result=_command_result(mod.live_model_import_command()),
            poison_test_text='assert artifact["fixture_cost_ratio"] > 1\n',
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_core_pretest_failed")
    assert artifact["core_pretest_green"] is False


def test_req_report_3914_import_failure_is_recorded_not_fatal(tmp_path: Path) -> None:
    """REQ-REPORT-3914: module import diagnostics are bare bools, not a hard gate."""

    _seed_repo(tmp_path)

    out_path = mod.run(
        tmp_path,
        summary_result=_summary_result(),
        poison_test_result=_command_result(mod.poison_test_command()),
        core_pretest_result=_command_result(mod.core_pretest_command()),
        live_model_import_result=_command_result(
            mod.live_model_import_command(),
            exit_code=1,
            stdout="ImportError\n",
        ),
        poison_test_text='assert artifact["fixture_cost_ratio"] > 1\n',
        started_s=8.0,
        now_s=9.0,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.terminal_verdict(live_model_modules_importable=False)
    assert artifact["live_model_modules_importable"] is False
    assert artifact["live_model_import_exit_code"] == 1
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["core_pretest_green"] is True


def test_req_report_3914_summary_parser_records_stamped_and_live_flags() -> None:
    """REQ-REPORT-3914: stamped and live-critical flags remain visible."""

    verdicts = mod.task_verdicts_from_summary(SUMMARY_STDOUT)

    assert verdicts["exp3903-archive-v360-activate-v361-green-gate"].startswith("complete:")
    assert verdicts["exp3904-moat-scissor-regated-accuracy-axis"] == "blocked_llama_cpp_inference_failed"
    assert "summarize_artifact LIVE_CRITICAL" in verdicts[
        "exp3905-build-test-cost-instrumented-verify-harness"
    ]
    assert "SKIP-cascaded by exp3905 poison-test pre-test failure" in verdicts[
        "exp3906-efficiency-head-to-head"
    ]


def test_req_report_3914_subprocess_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3914: subprocess helpers use the mandated commands."""

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
    assert mod.run_poison_test(tmp_path).exit_code == 0
    assert calls == [mod.poison_test_command()]

    calls.clear()
    assert mod.run_core_pretest(tmp_path).exit_code == 0
    assert calls == [mod.core_pretest_command()]

    calls.clear()
    assert mod.run_live_model_import_check(tmp_path).stdout == SUMMARY_STDOUT
    assert calls == [mod.live_model_import_command()]

    def fail_subprocess(cmd: list[str], **kwargs: object) -> object:
        raise mod.subprocess.CalledProcessError(1, cmd, output="", stderr="failed")

    monkeypatch.setattr(mod.subprocess, "run", fail_subprocess)
    assert mod.run_core_pretest(tmp_path).exit_code == 1
    assert mod.run_live_model_import_check(tmp_path).exit_code == 1

    def os_error_subprocess(cmd: list[str], **kwargs: object) -> object:
        raise OSError("missing executable")

    monkeypatch.setattr(mod.subprocess, "run", os_error_subprocess)
    assert mod.run_poison_test(tmp_path).exit_code == 127
    assert mod.run_core_pretest(tmp_path).exit_code == 127
    assert mod.run_live_model_import_check(tmp_path).exit_code == 127


def test_scenario_report_3914_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3914: the requested script entrypoint exists."""

    script = Path("scripts/experiments/experiment_3914_archive_v361_activate_v362.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "archive_v361_activate_v362_3914" in text


def test_req_report_3914_main_prints_written_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-3914: module main is a thin printable runner."""

    out_path = tmp_path / "results" / "experiment_3914_archive_v361_activate_v362.json"
    monkeypatch.setattr(mod, "run", lambda root: out_path)

    assert mod.main() == 0
    assert str(out_path) in capsys.readouterr().out
