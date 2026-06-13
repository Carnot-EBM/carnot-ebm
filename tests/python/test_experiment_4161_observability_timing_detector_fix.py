"""Tests for Exp 4161 operational-retro timing detector repair.

Spec refs: REQ-REPORT-4161, SCENARIO-REPORT-4161-FALLBACK,
SCENARIO-REPORT-4161-ROOT-CAUSE.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carnot import experiment_4161_observability_timing_detector_fix as runner_mod
from carnot.reporting import observability_timing_detector_4161 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
CHANGELOG_PATH = Path("ops/changelog.md")


def test_req_report_4161_spec_anchor_exists() -> None:
    """REQ-REPORT-4161: OpenSpec declares the fallback observability contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4161" in spec
    assert "SCENARIO-REPORT-4161-FALLBACK" in spec
    assert "SCENARIO-REPORT-4161-ROOT-CAUSE" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "false_zero_root_cause" in spec


def test_scenario_report_4161_root_cause_literal_exp_predicate_misses_titles() -> None:
    """SCENARIO-REPORT-4161-ROOT-CAUSE: title-only commits fail the legacy match."""

    subject = (
        "[conductor] ACCUMULATE pass 1 -- DIAGNOSE + FIX the max_epochs cap, "
        "then resume-train from the intact checkpoint"
    )

    assert mod.legacy_retro_subject_matches(subject) is False
    assert mod.legacy_retro_subject_matches("[conductor] Exp 4146: legacy title") is True
    assert "literal 'Exp '" in mod.FALSE_ZERO_ROOT_CAUSE
    assert "results/experiment_<digits>_*.json" in mod.FALSE_ZERO_ROOT_CAUSE


def test_scenario_report_4161_changelog_fallback_recovers_known_good_384_window() -> None:
    """SCENARIO-REPORT-4161-FALLBACK: .384 fallback returns non-zero evidence."""

    changelog_text = CHANGELOG_PATH.read_text(encoding="utf-8")

    detection = mod.detect_from_sources(
        "2026.06.384",
        git_log_text="",
        changelog_text=changelog_text,
        git_range="empty-range",
    )

    assert detection.experiment_count >= 10
    assert detection.fallback_used is True
    assert detection.source == mod.SOURCE_CHANGELOG_FALLBACK
    assert "results/experiment_4146_sudoku_accumulate_pass1_epochfix.json" in detection.artifact_paths
    assert "results/experiment_4155_capstone_v384.json" in detection.artifact_paths
    assert "results/experiment_4145_archive_v383_activate_v384.json" not in detection.artifact_paths
    assert "results/experiment_4160_arc_action_efficiency_harness.json" not in detection.artifact_paths


def test_req_report_4161_git_scan_wins_when_nonzero() -> None:
    """REQ-REPORT-4161: fallback is used only when git attribution is empty."""

    git_log_text = "\n".join(
        [
            "COMMIT\tabc\t2026-06-13T08:48:54-04:00\t[conductor] ACCUMULATE pass 1",
            "A\tresults/experiment_4146_sudoku_accumulate_pass1_epochfix.json",
        ]
    )

    detection = mod.detect_from_sources(
        "2026.06.384",
        git_log_text=git_log_text,
        changelog_text="",
        git_range="start..end",
    )

    assert detection.experiment_count == 1
    assert detection.fallback_used is False
    assert detection.source == mod.SOURCE_GIT_ATTRIBUTION
    assert detection.artifact_paths == (
        "results/experiment_4146_sudoku_accumulate_pass1_epochfix.json",
    )


def test_req_report_4161_git_parser_filters_auxiliary_and_modified_paths() -> None:
    """REQ-REPORT-4161: only added terminal experiment artifacts are counted."""

    git_log_text = "\n".join(
        [
            "COMMIT\tabc\t2026-06-13T08:48:54-04:00\t[conductor] title",
            "M\tresults/experiment_4146_modified.json",
            "A\tresults/experiment_4146_sudoku_accumulate_pass1_epochfix_state.json",
            "results/experiment_4146_sudoku_accumulate_pass1_epochfix.json",
            "A\tresults/experiment_not_numeric.json",
        ]
    )

    assert mod.extract_git_terminal_artifacts(git_log_text) == (
        "results/experiment_4146_sudoku_accumulate_pass1_epochfix.json",
    )
    assert mod.is_terminal_experiment_artifact(" results/experiment_1_ok.json ") is True
    assert mod.is_terminal_experiment_artifact("results/experiment_1_ok_state.json") is False


def test_req_report_4161_git_helpers_use_activation_bounds() -> None:
    """REQ-REPORT-4161: git helpers query activation commits and artifact names."""

    calls: list[list[str]] = []
    responses = {
        (
            "git",
            "log",
            "--format=%H%x09%aI%x09%s",
            "--grep=\\[conductor\\] Activate milestone 2026.06.384",
            "-n",
            "1",
        ): "start384\t2026-06-13T08:16:06-04:00\t[conductor] Activate milestone 2026.06.384\n",
        (
            "git",
            "log",
            "--format=%H%x09%aI%x09%s",
            "--grep=\\[conductor\\] Activate milestone 2026.06.385",
            "-n",
            "1",
        ): "start385\t2026-06-13T11:46:26-04:00\t[conductor] Activate milestone 2026.06.385\n",
        (
            "git",
            "log",
            "--format=COMMIT%x09%H%x09%aI%x09%s",
            "--name-status",
            "--diff-filter=A",
            "start384..start385",
            "--",
            "results/experiment_*.json",
        ): "A\tresults/experiment_4146_sudoku_accumulate_pass1_epochfix.json\n",
    }

    def runner(args: list[str], cwd: Path) -> str:
        calls.append(args)
        return responses[tuple(args)]

    assert mod.next_milestone("2026.06.384") == "2026.06.385"
    assert mod.activation_commit(Path("/repo"), "2026.06.384", run_git=runner) == "start384"
    assert mod.milestone_git_range(Path("/repo"), "2026.06.384", run_git=runner) == "start384..start385"
    assert (
        mod.git_log_name_status(Path("/repo"), "start384..start385", run_git=runner)
        == "A\tresults/experiment_4146_sudoku_accumulate_pass1_epochfix.json\n"
    )
    assert any("--diff-filter=A" in call for call in calls)

    assert mod.activation_commit(Path("/repo"), "2026.06.386", run_git=lambda *_: "") is None
    with pytest.raises(mod.DetectorPreconditionError, match="blocked_activation_commit"):
        mod.milestone_git_range(Path("/repo"), "2026.06.386", run_git=lambda *_: "")


def test_req_report_4161_payload_records_fix_and_fallback() -> None:
    """REQ-REPORT-4161: artifact fields are bare, terminal, and auditable."""

    changelog_text = CHANGELOG_PATH.read_text(encoding="utf-8")

    payload = mod.build_payload_from_text(
        milestone="2026.06.384",
        git_log_text="",
        changelog_text=changelog_text,
        git_range="empty-range",
        duration_s=0.1,
    )

    mod.validate_payload(payload)

    assert payload["honest_verdict"].startswith("complete:")
    assert payload["fix_applied"] is True
    assert payload["fallback_added"] is True
    assert payload["fallback_used"] is True
    assert payload["standalone_detector_module"] == "python/carnot/reporting/observability_timing_detector_4161.py"
    assert payload["research_conductor_touched"] is False
    assert payload["fallback_experiment_count"] >= 10
    assert "literal 'Exp '" in payload["false_zero_root_cause"]


def test_req_report_4161_payload_validation_rejects_bad_required_fields() -> None:
    """REQ-REPORT-4161: required artifact fields stay terminal and bare."""

    payload = mod.build_payload_from_text(
        milestone="2026.06.999",
        git_log_text="",
        changelog_text="",
        git_range="empty",
        duration_s=0.1,
    )
    assert payload["honest_verdict"] == "blocked_no_terminal_experiments_found"
    assert mod._duration_s(None) == 0.0001

    for mutate, expected in [
        (lambda p: p.pop("fix_applied"), "missing required"),
        (lambda p: p.__setitem__("honest_verdict", "not_terminal"), "terminal-prefixed"),
        (lambda p: p.__setitem__("fix_applied", "true"), "fix_applied"),
        (lambda p: p.__setitem__("fallback_added", 1), "fallback_added"),
        (lambda p: p.__setitem__("false_zero_root_cause", []), "false_zero_root_cause"),
    ]:
        bad = dict(payload)
        mutate(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_payload(bad)


def test_req_report_4161_build_payload_preconditions_and_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4161: blocked preconditions and write path are deterministic."""

    assert mod.build_payload(tmp_path)["honest_verdict"] == "blocked_retro_artifacts"

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "operational_retro_2026_06_384.json").write_text("{}", encoding="utf-8")
    assert mod.build_payload(tmp_path)["honest_verdict"] == "blocked_timing_detector_script"

    module_path = tmp_path / mod.STANDALONE_MODULE_PATH
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# detector exists\n", encoding="utf-8")
    assert mod.build_payload(tmp_path, run_git=lambda *_: "start\tdate\tsubject\n")["honest_verdict"] == "blocked_ops_changelog"

    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "changelog.md").write_text(
        "\n".join(
            [
                "- 2026-06-13: Archive .383 -> activate .384. results/experiment_4145_archive_v383_activate_v384.json",
                "- 2026-06-13: Work. results/experiment_4146_sudoku_accumulate_pass1_epochfix.json",
                "- 2026-06-13: Operational retrospective .384.",
            ]
        ),
        encoding="utf-8",
    )

    def runner(args: list[str], cwd: Path) -> str:
        if "2026.06.384" in " ".join(args):
            return "start384\t2026-06-13T08:16:06-04:00\tactivate\n"
        if "2026.06.385" in " ".join(args):
            return "start385\t2026-06-13T11:46:26-04:00\tactivate\n"
        return ""

    payload = mod.build_payload(tmp_path, run_git=runner, started_s=0.0)
    assert payload["honest_verdict"].startswith("complete:")
    assert payload["fallback_used"] is True
    assert payload["artifact_paths"] == [
        "results/experiment_4146_sudoku_accumulate_pass1_epochfix.json"
    ]

    written = mod.write_payload(tmp_path, payload)
    assert written == tmp_path / mod.OUTPUT_REL_PATH
    assert written.exists()

    monkeypatch.setattr(mod, "build_payload", lambda root: payload)
    assert mod.run(tmp_path) == written


def test_req_report_4161_experiment_entrypoint_prints_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-REPORT-4161: requested run command has a thin tested entrypoint."""

    output_path = tmp_path / mod.OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"honest_verdict": "complete: ok"}\n', encoding="utf-8")

    monkeypatch.setattr(runner_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner_mod, "run", lambda root: output_path)

    assert runner_mod.main() == 0
    assert '"honest_verdict": "complete: ok"' in capsys.readouterr().out
