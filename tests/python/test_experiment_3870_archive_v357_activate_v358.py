"""Tests for Exp 3870 .357 archive and .358 activation.

Spec refs: REQ-REPORT-3870, SCENARIO-REPORT-3870,
SCENARIO-REPORT-3870-BLOCKED-YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v357_activate_v358_3870 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


SUMMARY_STDOUT = """\
==============================================================================
ARTIFACT  experiment_3869_moat_scissor_v4_existing_corpus.json
------------------------------------------------------------------------------
  verdict          : complete: moat_scissor_v4_INCONCLUSIVE_reasoner_self_verify_auroc_and_carnot_ensemble_auroc
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  acceptance gates : (none found - claim has no self-reported gate)
  duration_s       : 64.00245833396912   substrate: live_llama_cpp_self_verification_plus_carnot_k15_ensemble_existing_prmbench_corpus
  headline metrics :
      carnot_ensemble_auroc = 0.551792
      reasoner_self_verify_auroc = 0.5
  adversarial flags:
      [info    ] IMPLAUSIBLE_PERFECT: n_reasoner_caught_errors=0.0 (exactly zero).
==============================================================================
"""


def _summary_result(exit_code: int = 0, stdout: str = SUMMARY_STDOUT) -> mod.SummaryResult:
    return mod.SummaryResult(
        command=[
            str(mod.PYTHON_BIN),
            "scripts/summarize_artifact.py",
            mod.EXP3869_ARTIFACT_REL_PATH.as_posix(),
        ],
        exit_code=exit_code,
        stdout=stdout,
        stderr="",
    )


def _seed_repo(root: Path, *, corrupt_complete: bool = False, milestone: str = "2026.06.358") -> None:
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{milestone}"\n'
        "tasks:\n"
        "  - id: exp3870-archive-v357-activate-v358-backend-diag\n"
        "    agent_type: codex\n"
        "    requires_codex: true\n",
        encoding="utf-8",
    )
    complete_text = (
        "milestones:\n"
        "- id: 2026.06.357\n"
        "  title: Verifier-MOAT-at-scale stale conductor archive\n"
        "  completed: '2026-06-05'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3869-moat-scissor-v4-against-existing-corpus\n"
        "    deliverable: results/experiment_3869_moat_scissor_v4_existing_corpus.json\n"
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


def test_req_report_3870_spec_anchor_exists() -> None:
    """REQ-REPORT-3870: OpenSpec declares the archive/activation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3870" in spec
    assert "SCENARIO-REPORT-3870" in spec
    assert "SCENARIO-REPORT-3870-BLOCKED-YAML" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.TERMINAL_VERDICT in spec


def test_scenario_report_3870_run_appends_v357_verdict_and_backend_diag(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3870: append .357 verdict and confirm .358 activation."""

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
        docs_gate_green=True,
        started_s=4.0,
        now_s=4.75,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    complete_text = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    complete = yaml.safe_load(complete_text)
    archived = complete["milestones"][-1]
    task_results = {task["id"]: task["result"] for task in archived["tasks"]}

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.TERMINAL_VERDICT
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["archived_milestone"] == "2026.06.357"
    assert artifact["activated_milestone"] == "2026.06.358"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["docs_gate_green"] is True
    assert artifact["active_milestone_confirmed"] is True
    assert artifact["active_roadmap_path"] == "research-roadmap.yaml"
    assert artifact["exp3869_honest_verdict"].startswith("complete: moat_scissor_v4_INCONCLUSIVE")
    assert artifact["exp3869_summary_exit_code"] == 0
    assert artifact["exp3869_summary_command"] == _summary_result().command
    assert "codex" in artifact["backend_routing_recommendation"]
    assert "gemini<->codex" in artifact["backend_routing_recommendation"]
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == 0.75
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    assert complete_text.startswith(before["complete"].rstrip())
    assert complete_text.count("correction_type: v357_inconclusive_archive_activation") == 1
    assert complete_text.count("- id: 2026.06.357") == 2
    assert "result: complete:" not in complete_text
    assert "result: 'complete: moat_scissor_v4_INCONCLUSIVE" in complete_text
    assert "result: 'COMPLETE: exp3870 archived .357 and activated .358" in complete_text
    assert "both positive controls degenerate" in archived["finding"]
    assert "out-of-distribution PRMBench" in archived["finding"]
    assert archived["activation_recorded"] == "exp3870-archive-v357-activate-v358"
    assert task_results["exp3869-moat-scissor-v4-against-existing-corpus"].startswith(
        "complete: moat_scissor_v4_INCONCLUSIVE"
    )
    assert task_results["exp3870-archive-v357-activate-v358"].startswith("COMPLETE:")
    assert yaml.safe_load(complete_text)

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before["roadmap"]
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before["changelog"]
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before["status"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8") == before["trace"]
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before["conductor"]


def test_req_report_3870_append_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3870: rerunning does not duplicate the archive record."""

    _seed_repo(tmp_path)

    first = mod.run(
        tmp_path,
        summary_result=_summary_result(),
        docs_gate_green=True,
        started_s=1.0,
        now_s=1.25,
    ).read_text(encoding="utf-8")
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second = mod.run(
        tmp_path,
        summary_result=_summary_result(),
        docs_gate_green=True,
        started_s=1.0,
        now_s=1.25,
    ).read_text(encoding="utf-8")
    second_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")

    assert first == second
    assert first_complete == second_complete
    assert second_complete.count("correction_type: v357_inconclusive_archive_activation") == 1


def test_scenario_report_3870_blocked_yaml_writes_artifact_without_append(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3870-BLOCKED-YAML: corrupt YAML exits before append."""

    _seed_repo(tmp_path, corrupt_complete=True)
    before = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")

    out_path = mod.run(
        tmp_path,
        summary_result=_summary_result(),
        docs_gate_green=True,
        started_s=7.0,
        now_s=7.1,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison")
    assert artifact["research_complete_yaml_parses"] is False
    assert artifact["preconditions_checked"]["research_complete_yaml_exists"] is True
    assert artifact["preconditions_checked"]["research_complete_yaml_parsed_before"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("archived_milestone"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(archived_milestone="2026.06.356"), "archived milestone"),
        (lambda p: p.update(activated_milestone="2026.06.357"), "activated milestone"),
        (lambda p: p.update(research_complete_yaml_parses=False), "YAML must parse"),
        (lambda p: p.update(active_milestone_confirmed=False), "active milestone"),
        (lambda p: p.update(docs_gate_green=False), "docs gate"),
        (lambda p: p.update(exp3869_honest_verdict="success: not inconclusive"), "Exp 3869"),
        (lambda p: p.update(backend_routing_recommendation="gemini only"), "backend"),
        (lambda p: p.update(inference_substrate="live_model"), "inference"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(copied_marker="CUDA"), "compute-bound markers"),
    ],
)
def test_req_report_3870_validate_artifact_rejects_regressions(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3870: validation rejects fields that hide archive risk."""

    _seed_repo(tmp_path)
    payload = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            docs_gate_green=True,
            started_s=9.0,
            now_s=9.5,
        ).read_text(encoding="utf-8")
    )

    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(broken)


def test_req_report_3870_helpers_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3870: helper failures block instead of fabricating success."""

    _seed_repo(tmp_path, milestone="2026.06.357")
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            docs_gate_green=True,
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_v358_not_active")

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(stdout="no verdict here"),
            docs_gate_green=True,
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_exp3869_summary_missing_verdict")

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(exit_code=2),
            docs_gate_green=True,
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_exp3869_summary_critical")

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            docs_gate_green=False,
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_docs_gate_failed")

    _seed_repo(tmp_path)
    before = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    monkeypatch.setattr(
        mod,
        "append_research_complete_record",
        lambda _text, _verdict: "milestones:\n- id: bad\n  result: complete: broken\n",
    )
    artifact = json.loads(
        mod.run(
            tmp_path,
            summary_result=_summary_result(),
            docs_gate_green=True,
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
            docs_gate_green=True,
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison_missing")


def test_req_report_3870_subprocess_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3870: subprocess helpers use summarizer and docs gate commands."""

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
    summary = mod.run_summarize_artifact(tmp_path)
    assert summary.stdout == SUMMARY_STDOUT
    assert summary.exit_code == 0
    assert summary.command == [
        str(mod.PYTHON_BIN),
        "scripts/summarize_artifact.py",
        mod.EXP3869_ARTIFACT_REL_PATH.as_posix(),
    ]

    calls.clear()
    assert mod.evaluate_docs_gate(tmp_path) is True
    assert calls == [[str(mod.PYTEST_BIN), "-o", "addopts=", "tests/python/test_docs.py", "-q"]]

    def fail_subprocess(cmd: list[str], **kwargs: object) -> object:
        raise mod.subprocess.CalledProcessError(1, cmd, output="", stderr="failed")

    monkeypatch.setattr(mod.subprocess, "run", fail_subprocess)
    assert mod.evaluate_docs_gate(tmp_path) is False

    def os_error_subprocess(cmd: list[str], **kwargs: object) -> object:
        raise OSError("missing pytest")

    monkeypatch.setattr(mod.subprocess, "run", os_error_subprocess)
    assert mod.evaluate_docs_gate(tmp_path) is False


def test_scenario_report_3870_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3870: the requested script entrypoint exists."""

    script = Path("scripts/experiments/experiment_3870_archive_v357_activate_v358.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "archive_v357_activate_v358_3870" in text
