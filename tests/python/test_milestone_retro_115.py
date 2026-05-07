"""Tests for the Exp 1505 milestone .115 retrospective.

Spec: REQ-REPORT-053, SCENARIO-REPORT-053.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import carnot.reporting.milestone_retro_115 as retro115
from carnot.reporting.milestone_retro_115 import (
    EXPECTED_EXPERIMENT_IDS,
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    _append_research_complete_archive,
    _load_sources,
    _number,
    _protected_files_clean,
    _read_json,
    _read_text,
    _research_complete_has_115_entry,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _terminal_sources() -> dict[str, dict[str, object]]:
    return {
        "exp1492": {
            "status": "complete",
            "milestone": "2026.04.115",
            "activation_manifest_complete": True,
            "guardrail_blocks_preserved": True,
            "honest_verdict": "complete: milestone_115_activation_complete",
        },
        "exp1493": {
            "status": "complete",
            "trigger_certificate_ready": True,
            "certificate_parse_rate": 0.3,
            "certificate_validation_rate": 0.1,
            "always_constrained_validation_rate": 0.0,
            "verifier_false_accept_rate": 0.0,
            "live_sota_model_inference_used": True,
            "honest_verdict": "complete: trigger certificates measured",
        },
        "exp1494": {
            "status": "complete",
            "validator_compiler_ready": True,
            "validator_compile_rate": 0.933333,
            "known_good_pass_rate": 1.0,
            "known_bad_reject_rate": 1.0,
            "verifier_false_accept_rate": 0.0,
            "arbitrary_code_execution_path_introduced": False,
            "honest_verdict": "complete: validator compiler measured",
        },
        "exp1495": {
            "status": "complete",
            "gated_inputs_present": True,
            "monitor_intervention_ready": True,
            "monitor_events_emitted": 40,
            "false_interruptions": 0,
            "verifier_false_accept_rate": 0.0,
            "honest_verdict": "complete: monitor replay ready",
        },
        "exp1496": {
            "status": "complete",
            "safe_prefix_continuation_ready": True,
            "safe_prefix_validator_pass_rate": 0.666667,
            "baseline_validator_pass_rate": 0.0,
            "full_regeneration_validator_pass_rate": 0.0,
            "verifier_false_accept_rate": 0.0,
            "honest_verdict": "complete: safe prefix improved",
        },
        "exp1497": {
            "status": "complete",
            "daily_eval_manifest_ready": True,
            "continuous_self_learning_task": "FR-11 v10 bounded trace2skill daily evaluation",
            "skills_evaluated": 24,
            "promoted_skill_count": 12,
            "retired_skill_count": 0,
            "soundness_mistakes": 0,
            "task_success_delta": 0.5,
            "honest_verdict": "complete: fr11 daily eval ready",
        },
        "exp1498": {
            "status": "complete",
            "artifact_reachability_audit_complete": True,
            "skills_checked": 24,
            "reachable_artifact_count": 2,
            "unreachable_artifact_count": 0,
            "ambiguous_resolver_count": 0,
            "repair_decisions": [],
            "retirement_decisions": [],
            "honest_verdict": "complete: reachability audit passed",
        },
        "exp1499": {
            "status": "complete",
            "orthogonality_matrix_written": True,
            "honest_verdict": "complete: orthogonality matrix written",
        },
        "exp1500": {
            "status": "complete",
            "discipline_gate_ready": True,
            "headline_allowed_signals": [
                "deterministic_executable_validators",
                "conservative_deterministic_bounds",
            ],
            "retired_signals": [
                "semantic_energy_headline_telemetry",
                "v1_pairwise_self_verification_active_gate",
            ],
            "honest_verdict": "complete: deterministic discipline ready",
        },
        "exp1501": {
            "status": "complete",
            "plan_graph_energy_ready": True,
            "graph_energy_beats_baselines": True,
            "node_localization_top1_rate": 1.0,
            "edge_localization_top1_rate": 1.0,
            "trained_gnn_used": False,
            "honest_verdict": "complete: plan graph energy ready",
        },
        "exp1502": {
            "status": "complete",
            "kan_hardware_accounting_ready": True,
            "accounting_only_no_synthesis_claim": True,
            "hardware_claim_allowed": False,
            "honest_verdict": "complete: kan accounting no hardware claim",
        },
        "exp1503": {
            "status": "complete",
            "thrml_import_ready": True,
            "parity_followup_allowed": True,
            "hardware_claim_allowed": False,
            "honest_verdict": "complete_thrml_import_ready",
        },
        "exp1504": {
            "status": "complete",
            "thrml_import_ready": True,
            "parity_experiment_ran": True,
            "parity_fail_count": 0,
            "parity_pass_count": 2,
            "simulator_only": True,
            "hardware_claim_allowed": False,
            "honest_verdict": "complete_thrml_parity_passed_no_hardware_claim",
        },
    }


def _conductor_log_text() -> str:
    titles = [
        ".114 Completion Archive + .115 Activation Manifest",
        "Trigger-Token Certificate Export v1",
        "ConstrainPrompt Validator Compiler Audit",
        "interwhen Monitor Prototype",
        "HoVer Safe-Prefix Continuation Audit",
        "FR-11 v10 Trace2Skill Daily Eval + Rot Check",
        "trace2skill Artifact Reachability Audit",
        "Verifier Ensemble DRY + Conditional Orthogonality",
        "Latent-vs-Deterministic Discipline Gate",
        "GNNVerifier Plan-Graph Energy Adapter Smoke",
        "KAN Hardware Accounting",
        "THRML Import Readiness Repair + Terminal Gate",
        "THRML/Carnot Simulator Parity v3",
    ]
    return "\n".join(
        f"| 2026-05-07 18:{idx:02d} UTC | {title} | OK | 81 passed |"
        for idx, title in enumerate(titles, start=1)
    )


def test_req_report_053_scores_all_115_criteria_and_claim_decisions() -> None:
    """REQ-REPORT-053: .115 criteria and claim-boundary decisions use source fields."""

    artifact = build_artifact(
        sources=_terminal_sources(),
        missing_source_ids=[],
        conductor_log_text=_conductor_log_text(),
        roadmap_doc_text="Target threshold: at least 11 of 14 tasks complete.",
        ops_status_text="Milestone .115 planning complete.",
        ops_changelog_text="Milestone .115 planned.",
        known_issues_text="No new mandatory blocker.",
        protected_files_unchanged=True,
        research_complete_updated=True,
        research_complete_update_reason="appended_2026.04.115_archive_entry",
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.115"
    assert artifact["criteria_met"] == 12
    assert artifact["criteria_total"] == 12
    assert artifact["experiments_reviewed"] == list(EXPECTED_EXPERIMENT_IDS)
    assert artifact["completed_experiments"] == list(EXPECTED_EXPERIMENT_IDS)
    assert artifact["honest_gate_skips"] == []
    assert artifact["ops_docs_updated"] is False
    assert artifact["research_complete_updated"] is True
    assert artifact["protected_files_unchanged"] is True
    assert artifact["success_criteria_results"]["thrml_readiness"]["status"] == "met"
    assert artifact["success_criteria_results"]["retrospective"]["status"] == "met"
    assert artifact["line_decisions"]["semantic_energy_v1"]["decision"] == "retired"
    assert artifact["line_decisions"]["fr11_trace2skill"]["decision"] == "graduated"
    assert artifact["line_decisions"]["kan_hardware_accounting"]["decision"] == "carry_forward"
    assert "bounded FR-11 daily evaluation" in artifact["continuous_self_learning_outcome"]
    assert "no Extropic TSU" in artifact["hardware_claim_boundaries"]
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_report_053_records_gate_skip_without_fabricating_success() -> None:
    """SCENARIO-REPORT-053: structured skips can satisfy gated criteria without fake runs."""

    sources = _terminal_sources()
    sources["exp1503"] = {
        "status": "complete",
        "thrml_import_ready": False,
        "parity_followup_allowed": False,
        "hardware_claim_allowed": False,
        "honest_verdict": "complete_thrml_import_not_ready_terminal",
    }
    sources["exp1504"] = {
        "status": "skipped",
        "gated_skip": True,
        "gated_off_reason": "exp1503.thrml_import_ready was false",
        "thrml_import_ready": False,
        "parity_experiment_ran": False,
        "hardware_claim_allowed": False,
        "honest_verdict": "complete_thrml_parity_structured_gate_skip",
    }

    artifact = build_artifact(
        sources=sources,
        missing_source_ids=[],
        conductor_log_text=_conductor_log_text(),
        roadmap_doc_text="THRML/Carnot parity only runs if exp1503 is true.",
        ops_status_text="",
        ops_changelog_text="",
        known_issues_text="",
        protected_files_unchanged=True,
        research_complete_updated=False,
        research_complete_update_reason="not_written_in_unit_test",
    )

    assert artifact["criteria_total"] == 12
    assert artifact["criteria_met"] == 12
    assert artifact["success_criteria_results"]["thrml_readiness"]["status"] == "met"
    assert artifact["honest_gate_skips"] == [
        {
            "experiment_id": "exp1504",
            "criterion": "thrml_readiness",
            "reason": "exp1503.thrml_import_ready was false",
        }
    ]
    assert "exp1504" not in artifact["completed_experiments"]


def test_req_report_053_run_writes_artifact_and_research_complete_archive(tmp_path: Path) -> None:
    """REQ-REPORT-053: run writes bootstrap, terminal JSON, and the .115 archive row."""

    out_path = tmp_path / "results" / "experiment_1505_milestone_115_retro.json"
    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    for exp_id, filename in SOURCE_FILES.items():
        _write_json(tmp_path / "results" / filename, _terminal_sources()[exp_id])
    (tmp_path / "ops").mkdir(exist_ok=True)
    (tmp_path / "ops" / "conductor-log.md").write_text(
        _conductor_log_text(),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "status.md").write_text("status evidence", encoding="utf-8")
    (tmp_path / "ops" / "changelog.md").write_text("changelog evidence", encoding="utf-8")
    (tmp_path / "ops" / "known-issues.md").write_text("known issues", encoding="utf-8")
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "Success Criteria",
        encoding="utf-8",
    )
    (tmp_path / "research-complete.yaml").write_text(
        "- id: 2026.04.114\n  title: prior\n",
        encoding="utf-8",
    )

    artifact = run(root=tmp_path, out_path=out_path, protected_files_unchanged=True)

    written = json.loads(out_path.read_text(encoding="utf-8"))
    archive = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))
    archive_115 = [entry for entry in archive if entry.get("id") == "2026.04.115"]

    assert artifact == written
    assert written["status"] == "complete"
    assert written["research_complete_updated"] is True
    assert len(archive_115) == 1
    assert archive_115[0]["tasks"][-1]["id"] == "exp1505-milestone-115-retrospective"
    assert archive_115[0]["tasks"][-1]["deliverable"] == (
        "results/experiment_1505_milestone_115_retro.json"
    )
    assert "ops_docs_updated" in written


def test_req_report_053_defensive_branches_stay_explicit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """REQ-REPORT-053: missing inputs and archive guards do not fabricate closure."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _number("not-a-number") is None
    assert _load_sources(tmp_path / "empty-results")[1] == list(EXPECTED_EXPERIMENT_IDS)

    sources = _terminal_sources()
    for exp_id in ("exp1492", "exp1499", "exp1503"):
        sources.pop(exp_id)
    artifact = build_artifact(
        sources=sources,
        missing_source_ids=["exp1492", "exp1499", "exp1503"],
        conductor_log_text="",
        roadmap_doc_text="",
        ops_status_text="",
        ops_changelog_text="",
        known_issues_text="",
        protected_files_unchanged=False,
        research_complete_updated=False,
        research_complete_update_reason="not_written_in_unit_test",
    )
    assert artifact["criteria_met"] < artifact["criteria_total"]
    assert artifact["experiment_verdicts"]["exp1492"]["status"] == "missing"
    assert artifact["success_criteria_results"]["verifier_discipline"]["status"] == "unmet"
    assert artifact["success_criteria_results"]["thrml_readiness"]["status"] == "unmet"

    sources_with_unknown_thrml = _terminal_sources()
    sources_with_unknown_thrml["exp1503"]["thrml_import_ready"] = None
    unknown_thrml = build_artifact(
        sources=sources_with_unknown_thrml,
        missing_source_ids=[],
        conductor_log_text="",
        roadmap_doc_text="",
        ops_status_text="",
        ops_changelog_text="",
        known_issues_text="",
        protected_files_unchanged=True,
        research_complete_updated=False,
        research_complete_update_reason="not_written_in_unit_test",
    )
    assert unknown_thrml["success_criteria_results"]["thrml_readiness"]["status"] == "unmet"

    assert not _research_complete_has_115_entry("")
    assert _research_complete_has_115_entry("id: 2026.04.115\n")

    original_safe_load = yaml.safe_load

    def raise_yaml_error(_text):
        raise yaml.YAMLError("bad yaml")

    monkeypatch.setattr(retro115.yaml, "safe_load", raise_yaml_error)
    assert _research_complete_has_115_entry("contains 2026.04.115 anyway")
    monkeypatch.setattr(retro115.yaml, "safe_load", original_safe_load)

    missing_archive = tmp_path / "missing-research-complete.yaml"
    assert _append_research_complete_archive(missing_archive, artifact) == (
        False,
        "research_complete_yaml_missing",
    )
    archive_path = tmp_path / "research-complete.yaml"
    archive_path.write_text("- id: 2026.04.115\n  title: present\n", encoding="utf-8")
    assert _append_research_complete_archive(archive_path, artifact) == (
        False,
        "research_complete_already_contains_2026.04.115",
    )
    archive_path.write_text("- id: 2026.04.114\n  title: prior\n", encoding="utf-8")
    incomplete = dict(artifact)
    incomplete["status"] = "blocked"
    assert _append_research_complete_archive(archive_path, incomplete) == (
        False,
        "terminal_retro_artifact_not_complete",
    )

    class CleanResult:
        returncode = 0

    class DirtyResult:
        returncode = 1

    monkeypatch.setattr(retro115.subprocess, "run", lambda *args, **kwargs: CleanResult())
    assert _protected_files_clean(tmp_path) is True
    monkeypatch.setattr(retro115.subprocess, "run", lambda *args, **kwargs: DirtyResult())
    assert _protected_files_clean(tmp_path) is False

    def raise_os_error(*_args, **_kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(retro115.subprocess, "run", raise_os_error)
    assert _protected_files_clean(tmp_path) is True
