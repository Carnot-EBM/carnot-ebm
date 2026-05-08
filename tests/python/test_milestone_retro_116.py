"""Tests for the Exp 1518 milestone .116 retrospective.

Spec: REQ-REPORT-055, SCENARIO-REPORT-055.
"""

from __future__ import annotations

import json
from pathlib import Path

import carnot.reporting.milestone_retro_116 as retro116
from carnot.reporting.milestone_retro_116 import (
    EXPECTED_EXPERIMENT_IDS,
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    _load_sources,
    _protected_files_clean,
    _read_json,
    _read_text,
    _research_complete_has_116_entry,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _terminal_sources() -> dict[str, dict[str, object]]:
    return {
        "exp1506": {
            "status": "complete",
            "activation_manifest_complete": True,
            "protected_files_unchanged": True,
            "research_roadmap_yaml_modified": False,
            "scripts_research_conductor_modified": False,
            "honest_verdict": "complete: milestone_116_activation_complete",
        },
        "exp1507": {
            "status": "complete",
            "verifier_induction_ready": True,
            "candidate_verifiers_compiled": 2,
            "verifier_compile_rate": 1.0,
            "verifier_coverage_rate": 1.0,
            "verifier_false_accept_rate": 0.0,
            "honest_verdict": "complete: safe dsl induction ready",
        },
        "exp1508": {
            "status": "complete",
            "certificate_decoder_ready": True,
            "live_sota_model_inference_used": True,
            "grammar_parse_rate": 1.0,
            "grammar_validation_rate": 1.0,
            "verifier_false_accept_rate": 0.0,
            "honest_verdict": "complete: grammar decoder ready",
        },
        "exp1509": {
            "status": "complete",
            "monitor_runtime_ready": True,
            "events_normalized": 60,
            "verifier_false_accept_rate": 0.0,
            "honest_verdict": "complete: runtime monitor ready",
        },
        "exp1510": {
            "status": "complete",
            "structural_contract_gate_ready": True,
            "violations_detected": 60,
            "false_accept_rate": 0.0,
            "honest_verdict": "complete: structural contracts ready",
        },
        "exp1511": {
            "status": "complete",
            "product_line_benchmark_ready": True,
            "solver_oracle_ready": True,
            "feasibility_rate": 0.0,
            "verifier_false_accept_rate": 0.0,
            "honest_verdict": "complete: solver oracle ready",
        },
        "exp1512": {
            "status": "complete",
            "policy_cache_ready": True,
            "continuous_self_learning_task": True,
            "no_model_weight_mutation": True,
            "soundness_mistakes": 0,
            "verifier_false_accept_rate": 0.0,
            "honest_verdict": "complete: policy cache ready",
        },
        "exp1513": {
            "status": "complete",
            "rollback_audit_passed": True,
            "accepted_policy_updates": 84,
            "false_accept_delta": 0,
            "soundness_mistakes": 0,
            "honest_verdict": "complete: rollback passed",
        },
        "exp1514": {
            "status": "complete",
            "portable_skill_pack_ready": True,
            "rollback_passing_entries": 24,
            "packaged_skill_entries": 24,
            "rejected_skill_entries": 0,
            "honest_verdict": "complete: portable pack ready",
        },
        "exp1515": {
            "status": "complete",
            "thrml_samplerbackend_conformance_ready": True,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "honest_verdict": "complete_thrml_conformance_ready_no_hardware",
        },
        "exp1516": {
            "status": "complete",
            "kan_shape_manifest_ready": True,
            "no_synthesis_claim": True,
            "no_board_claim": True,
            "honest_verdict": "complete: kan shape accounting ready",
        },
        "exp1517": {
            "status": "complete",
            "kv260_property_pack_ready": True,
            "source_level_only": True,
            "no_board_execution": True,
            "no_bitstream_claim": True,
            "honest_verdict": "complete: kv260 source properties ready",
        },
    }


def test_req_report_055_scores_all_116_criteria_and_claim_boundaries() -> None:
    """REQ-REPORT-055: .116 criteria and retirements use source artifact fields."""

    artifact = build_artifact(
        sources=_terminal_sources(),
        missing_source_ids=[],
        conductor_log_text="| 2026-05-08 UTC | Product-Line Solver Oracle Benchmark | OK |",
        roadmap_doc_text="Target threshold: at least 11 of 13 tasks complete.",
        research_roadmap_yaml_text="milestone: 2026.04.116",
        research_complete_text="- id: 2026.04.115\n  title: prior\n",
        ops_status_text="status evidence",
        ops_changelog_text="changelog evidence",
        protected_files_unchanged=True,
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.116"
    assert artifact["criteria_met"] == 13
    assert artifact["criteria_total"] == 13
    assert artifact["completed_tasks"] == [*EXPECTED_EXPERIMENT_IDS, "exp1518"]
    assert artifact["gated_or_blocked_tasks"] == []
    assert artifact["failed_tasks"] == []
    assert artifact["verifier_runtime_contract_ready"] is True
    assert "policy cache" in artifact["continuous_self_learning_result"]
    assert "no model-weight mutation" in artifact["continuous_self_learning_result"]
    assert "software/source conformance" in artifact["substrate_conformance_result"]
    assert artifact["ops_docs_updated"] is False
    assert artifact["research_complete_entry_recommended"]["entry"]["id"] == "2026.04.116"
    assert artifact["research_complete_entry_recommended"]["written"] is False
    assert artifact["protected_file_modification_findings"]["any_modification_reported"] is False
    assert "Semantic Energy/logit telemetry" in {
        claim["claim"] for claim in artifact["retired_or_demoted_claims"]
    }
    assert artifact["success_criteria_results"]["retrospective"]["status"] == "met"
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_report_055_gate_blocked_thrml_is_not_failure() -> None:
    """SCENARIO-REPORT-055: an honest simulator-only THRML blocker is not a failure."""

    sources = _terminal_sources()
    sources["exp1515"] = {
        "status": "blocked",
        "thrml_samplerbackend_conformance_ready": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "blockers": "thrml_runtime_unavailable",
        "honest_verdict": "complete: simulator_only_thrml_blocker_no_hardware_claim",
    }

    artifact = build_artifact(
        sources=sources,
        missing_source_ids=[],
        conductor_log_text="",
        roadmap_doc_text="THRML conformance accepts an honest simulator-only blocker.",
        research_roadmap_yaml_text="",
        research_complete_text="",
        ops_status_text="",
        ops_changelog_text="",
        protected_files_unchanged=True,
    )

    assert artifact["criteria_met"] == 13
    assert artifact["success_criteria_results"]["thrml_conformance"]["status"] == "gate_blocked"
    assert artifact["gated_or_blocked_tasks"] == [
        {
            "experiment_id": "exp1515",
            "criterion": "thrml_conformance",
            "reason": "thrml_runtime_unavailable",
        }
    ]
    assert artifact["failed_tasks"] == []

    sources["exp1515"]["blockers"] = ["thrml_runtime_unavailable"]
    list_blocker = build_artifact(
        sources=sources,
        missing_source_ids=[],
        conductor_log_text="",
        roadmap_doc_text="",
        research_roadmap_yaml_text="",
        research_complete_text="",
        ops_status_text="",
        ops_changelog_text="",
        protected_files_unchanged=True,
    )
    assert list_blocker["gated_or_blocked_tasks"][0]["reason"] == "thrml_runtime_unavailable"

    sources["exp1510"] = {
        "status": "complete",
        "structural_contract_gate_ready": False,
        "random_baseline_detection_rate": 0.0,
        "length_baseline_detection_rate": 0.0,
        "honest_verdict": "complete: structural_contract_no-signal_terminal",
    }
    no_signal = build_artifact(
        sources=sources,
        missing_source_ids=[],
        conductor_log_text="",
        roadmap_doc_text="",
        research_roadmap_yaml_text="",
        research_complete_text="",
        ops_status_text="",
        ops_changelog_text="",
        protected_files_unchanged=True,
    )
    assert no_signal["success_criteria_results"]["structural_contracts"]["status"] == "met"


def test_req_report_055_run_writes_terminal_json_without_ops_or_archive_mutation(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-055: run writes the JSON and recommends, but does not append, archive data."""

    out_path = tmp_path / "results" / "experiment_1518_milestone_116_retro.json"
    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    for exp_id, filename in SOURCE_FILES.items():
        _write_json(tmp_path / "results" / filename, _terminal_sources()[exp_id])
    (tmp_path / "ops").mkdir(exist_ok=True)
    (tmp_path / "ops" / "conductor-log.md").write_text("conductor evidence", encoding="utf-8")
    (tmp_path / "ops" / "status.md").write_text("status evidence", encoding="utf-8")
    (tmp_path / "ops" / "changelog.md").write_text("changelog evidence", encoding="utf-8")
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "Success Criteria",
        encoding="utf-8",
    )
    (tmp_path / "research-roadmap.yaml").write_text(
        "milestone: 2026.04.116\n",
        encoding="utf-8",
    )
    research_complete_path = tmp_path / "research-complete.yaml"
    research_complete_path.write_text("- id: 2026.04.115\n  title: prior\n", encoding="utf-8")

    artifact = run(root=tmp_path, out_path=out_path, protected_files_unchanged=True)

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact == written
    assert written["status"] == "complete"
    assert written["criteria_met"] == 13
    assert written["ops_docs_updated"] is False
    assert "separate_reconciliation_agent" in written["ops_docs_update_deferred_reason"]
    assert research_complete_path.read_text(encoding="utf-8") == (
        "- id: 2026.04.115\n  title: prior\n"
    )
    assert written["research_complete_entry_recommended"]["written"] is False


def test_req_report_055_defensive_branches_stay_explicit(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-055: missing inputs and protected-file reports do not fake closure."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _load_sources(tmp_path / "empty-results")[1] == list(EXPECTED_EXPERIMENT_IDS)
    assert not _research_complete_has_116_entry("")
    assert _research_complete_has_116_entry("- id: 2026.04.116\n")

    sources = _terminal_sources()
    sources["exp1506"]["research_roadmap_yaml_modified"] = True
    sources["exp1506"]["scripts_research_conductor_modified"] = True
    sources.pop("exp1511")
    artifact = build_artifact(
        sources=sources,
        missing_source_ids=["exp1511"],
        conductor_log_text="",
        roadmap_doc_text="",
        research_roadmap_yaml_text="",
        research_complete_text="id: 2026.04.116\n",
        ops_status_text="",
        ops_changelog_text="",
        protected_files_unchanged=False,
    )

    assert artifact["criteria_met"] < artifact["criteria_total"]
    assert artifact["success_criteria_results"]["feature_model_oracle"]["status"] == "unmet"
    assert artifact["success_criteria_results"]["retrospective"]["status"] == "unmet"
    assert artifact["failed_tasks"][0]["experiment_id"] == "exp1511"
    assert artifact["protected_file_modification_findings"]["any_modification_reported"] is True
    assert artifact["research_complete_entry_recommended"]["already_present"] is True

    class CleanResult:
        returncode = 0

    class DirtyResult:
        returncode = 1

    monkeypatch.setattr(retro116.subprocess, "run", lambda *args, **kwargs: CleanResult())
    assert _protected_files_clean(tmp_path) is True
    monkeypatch.setattr(retro116.subprocess, "run", lambda *args, **kwargs: DirtyResult())
    assert _protected_files_clean(tmp_path) is False

    def raise_os_error(*_args, **_kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(retro116.subprocess, "run", raise_os_error)
    assert _protected_files_clean(tmp_path) is True
