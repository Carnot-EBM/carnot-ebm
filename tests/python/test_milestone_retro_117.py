"""Tests for the Exp 1532 milestone .117 retrospective.

Spec: REQ-REPORT-057, SCENARIO-REPORT-057.
"""

from __future__ import annotations

import json
from pathlib import Path

import carnot.reporting.milestone_retro_117 as retro117
from carnot.reporting.milestone_retro_117 import (
    EXPECTED_EXPERIMENT_IDS,
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    _blocker_reason,
    _load_sources,
    _protected_files_clean,
    _read_json,
    _read_text,
    _research_complete_has_117_entry,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _terminal_sources() -> dict[str, dict[str, object]]:
    return {
        "exp1519": {
            "status": "complete",
            "activation_manifest_complete": True,
            "research_roadmap_yaml_modified": False,
            "scripts_research_conductor_modified": False,
            "honest_verdict": "complete: activation ready",
        },
        "exp1520": {
            "status": "complete",
            "runtime_contract_e2e_ready": True,
            "source_artifacts_loaded": True,
            "contract_cases_total": 458,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
            "honest_verdict": "complete: runtime contract ready",
        },
        "exp1521": {
            "status": "complete",
            "contract_guided_repair_ready": True,
            "live_sota_model_inference_used": True,
            "models_used": ["unsloth/Qwen3.6-35B-A3B-GGUF"],
            "repair_cases_attempted": 2,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
            "honest_verdict": "complete: repair ready",
        },
        "exp1522": {
            "status": "complete",
            "cdg_root_cause_repair_ready": True,
            "root_cause_cases_attempted": 111,
            "cdg_efficiency_delta": 0.05015,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
            "honest_verdict": "complete: cdg ready",
        },
        "exp1523": {
            "status": "complete",
            "product_line_rescue_ready": True,
            "product_line_branch_retired": False,
            "baseline_parse_rate": 0.333333,
            "rescue_parse_rate": 1.0,
            "baseline_oracle_agreement_rate": 0.0,
            "rescue_oracle_agreement_rate": 1.0,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
            "honest_verdict": "complete: product-line rescued",
        },
        "exp1524": {
            "status": "complete",
            "live_policy_promotion_ready": True,
            "continuous_self_learning_task": True,
            "no_model_weight_mutation": True,
            "rollback_passing_updates_loaded": 24,
            "soundness_mistakes": 0,
            "utility_delta": 0.0,
            "honest_verdict": "complete: fr11 promoted",
        },
        "exp1525": {
            "status": "complete",
            "claim_isolation_ablation_ready": True,
            "cases_loaded": 1,
            "claims_extracted": 4,
            "budget_delta": 3,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
            "honest_verdict": "complete: claim isolation ready",
        },
        "exp1526": {
            "status": "complete",
            "thrml_parity_n8_passed": True,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "n_spins": 8,
            "kl_divergence": 0.0,
            "honest_verdict": "complete_thrml_n8",
        },
        "exp1527": {
            "status": "complete",
            "thrml_parity_n16_passed": True,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "n_spins": 16,
            "kl_divergence": 0.0,
            "honest_verdict": "complete_thrml_n16",
        },
        "exp1528": {
            "status": "complete",
            "thrml_parity_n32_passed": True,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "n_spins": 32,
            "kl_divergence": 0.0,
            "honest_verdict": "complete_thrml_n32",
        },
        "exp1529": {
            "status": "complete",
            "thrml_parity_n64_passed": True,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "n_spins": 64,
            "kl_divergence": 0.0,
            "honest_verdict": "complete_thrml_n64",
        },
        "exp1530": {
            "status": "complete",
            "thrml_parity_n128_passed": True,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "n_spins": 128,
            "n_samples_per_backend": 10240,
            "kl_divergence": 0.0,
            "honest_verdict": "complete_thrml_n128",
        },
        "exp1531": {
            "status": "complete",
            "diverse_topology_parity_ready": True,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "topologies_tested": ["complete", "sparse_random", "lattice", "scale_free"],
            "topologies_passed": ["complete", "sparse_random", "lattice", "scale_free"],
            "kl_divergence_by_topology": {
                "complete": 0.0,
                "sparse_random": 0.0,
                "lattice": 0.0,
                "scale_free": 0.0,
            },
            "honest_verdict": "complete_thrml_diverse",
        },
    }


def test_req_report_057_scores_all_117_criteria_and_carry_forward_gates() -> None:
    """REQ-REPORT-057: .117 criteria and carry-forward gates use source fields."""

    artifact = build_artifact(
        sources=_terminal_sources(),
        missing_source_ids=[],
        conductor_log_text="| 2026-05-08 UTC | THRML Diverse Topology Parity n=32 | OK |",
        roadmap_doc_text="Success Criteria\nTarget threshold: at least 12 of 14",
        research_roadmap_yaml_text="milestone: 2026.04.117\n",
        research_complete_text="- id: 2026.04.116\n",
        ops_status_text="status evidence",
        ops_changelog_text="changelog evidence",
        protected_files_unchanged=True,
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.117"
    assert artifact["criteria_met"] == 14
    assert artifact["criteria_total"] == 14
    assert {result["status"] for result in artifact["criteria_results"].values()} == {"MET"}
    assert artifact["runtime_contract_e2e_outcome"]["false_accept_rate"] == 0.0
    assert artifact["live_contract_repair_outcome"]["mandated_sota_used"] is True
    assert artifact["cdg_root_cause_outcome"]["cdg_efficiency_delta"] == 0.05015
    assert artifact["product_line_decision"]["decision"] == "continue"
    assert artifact["product_line_decision"]["rescue_oracle_agreement_rate"] == 1.0
    assert artifact["continuous_self_learning_outcome"]["no_model_weight_mutation"] is True
    assert artifact["claim_isolation_outcome"]["deterministic_validators_final_authority"] is True
    assert artifact["thrml_scaling_outcome"]["next_scaling_gate"]["artifact_fields"] == [
        "thrml_parity_n256_passed",
        "diverse_topology_parity_n64_ready",
        "simulator_only",
        "no_tsu_hardware_claim",
    ]
    assert artifact["claim_boundaries_preserved"] is True
    assert {check["boundary"] for check in artifact["claim_boundary_checks"]} == {
        "no_tsu_hardware_claim",
        "no_kan_synthesis_claim",
        "no_kv260_board_claim",
        "no_arbitrary_generated_python_trust",
        "no_legacy_small_model_headline_result",
        "no_llm_judge_final_authority",
    }
    assert artifact["ops_docs_reconciled"] is False
    assert "separate_reconciliation_agent" in artifact["ops_docs_reconciliation_deferred_reason"]
    assert artifact["research_complete_entry_recommended"]["written"] is False
    assert artifact["research_complete_entry_recommended"]["entry"]["id"] == "2026.04.117"
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_report_057_retirement_and_gate_blockers_stay_explicit() -> None:
    """SCENARIO-REPORT-057: retired and gate-blocked branches are not fabricated wins."""

    sources = _terminal_sources()
    sources["exp1523"].update(
        {
            "product_line_rescue_ready": False,
            "product_line_branch_retired": True,
            "rescue_parse_rate": 0.333333,
            "rescue_oracle_agreement_rate": 0.0,
            "honest_verdict": "complete: product-line retired after no useful signal",
        }
    )
    sources["exp1530"] = {
        "status": "blocked",
        "thrml_parity_n128_passed": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "blockers": ["thrml_runtime_unavailable"],
        "honest_verdict": "complete: simulator-only blocker no hardware claim",
    }

    artifact = build_artifact(
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

    assert artifact["criteria_results"]["product_line_rescue"]["status"] == "MET"
    assert artifact["product_line_decision"]["decision"] == "retire"
    assert artifact["criteria_results"]["thrml_n128"]["status"] == "GATE_BLOCKED"
    assert artifact["gated_or_blocked_tasks"] == [
        {
            "experiment_id": "exp1530",
            "criterion": "thrml_n128",
            "reason": "thrml_runtime_unavailable",
        }
    ]
    assert artifact["failed_tasks"] == []

    sources.pop("exp1521")
    sources["exp1519"]["research_roadmap_yaml_modified"] = True
    sources["exp1519"]["scripts_research_conductor_modified"] = True
    missing = build_artifact(
        sources=sources,
        missing_source_ids=["exp1521"],
        conductor_log_text="",
        roadmap_doc_text="",
        research_roadmap_yaml_text="",
        research_complete_text="id: 2026.04.117\n",
        ops_status_text="",
        ops_changelog_text="",
        protected_files_unchanged=False,
    )
    assert missing["criteria_results"]["live_contract_repair"]["status"] == "NOT_MET"
    assert missing["criteria_results"]["retrospective"]["status"] == "NOT_MET"
    assert missing["failed_tasks"][0]["experiment_id"] == "exp1521"
    assert missing["protected_file_modification_findings"]["any_modification_reported"] is True
    assert missing["protected_file_modification_findings"]["source_reports"] == [
        {"experiment_id": "exp1519", "file": "research-roadmap.yaml"},
        {"experiment_id": "exp1519", "file": "scripts/research_conductor.py"},
    ]
    assert missing["research_complete_entry_recommended"]["already_present"] is True


def test_req_report_057_run_writes_terminal_json_without_ops_or_archive_mutation(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-057: run writes bootstrap and final JSON without archive mutation."""

    out_path = tmp_path / "results" / "experiment_1532_milestone_117_retro.json"
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
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.04.117\n", encoding="utf-8")
    research_complete_path = tmp_path / "research-complete.yaml"
    research_complete_path.write_text("- id: 2026.04.116\n  title: prior\n", encoding="utf-8")

    artifact = run(root=tmp_path, out_path=out_path, protected_files_unchanged=True)

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact == written
    assert written["status"] == "complete"
    assert written["criteria_met"] == 14
    assert written["ops_docs_reconciled"] is False
    assert research_complete_path.read_text(encoding="utf-8") == (
        "- id: 2026.04.116\n  title: prior\n"
    )
    assert written["research_complete_entry_recommended"]["written"] is False


def test_req_report_057_defensive_helpers_stay_explicit(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-057: missing files and protected-file checks stay explicit."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _load_sources(tmp_path / "empty-results")[1] == list(EXPECTED_EXPERIMENT_IDS)
    assert not _research_complete_has_117_entry("")
    assert _research_complete_has_117_entry("- id: 2026.04.117\n")
    assert _blocker_reason({"blockers": "plain blocker"}) == "plain blocker"
    assert _blocker_reason({"honest_verdict": "fallback verdict"}) == "fallback verdict"

    class CleanResult:
        returncode = 0

    class DirtyResult:
        returncode = 1

    monkeypatch.setattr(retro117.subprocess, "run", lambda *args, **kwargs: CleanResult())
    assert _protected_files_clean(tmp_path) is True
    monkeypatch.setattr(retro117.subprocess, "run", lambda *args, **kwargs: DirtyResult())
    assert _protected_files_clean(tmp_path) is False

    def raise_os_error(*_args, **_kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(retro117.subprocess, "run", raise_os_error)
    assert _protected_files_clean(tmp_path) is True
