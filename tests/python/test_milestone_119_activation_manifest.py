"""Tests for the Exp 1547 `.119` activation manifest.

Spec: REQ-REPORT-060, SCENARIO-REPORT-060.
"""

from __future__ import annotations

import json
from pathlib import Path

import carnot.reporting.milestone_119_activation_manifest as activation119
from carnot.reporting.milestone_119_activation_manifest import (
    ALLOWED_119_TRACKS,
    MANDATED_SOTA_MODELS,
    REQUIRED_ARTIFACT_FIELDS,
    RETIRED_HEADLINE_SIGNALS,
    SOURCE_FILES,
    _load_sources,
    _protected_files_clean,
    _read_json,
    _read_text,
    _relative_path,
    _research_complete_has_118_entry,
    _retirement_blocks_recorded,
    _thrml_independent_rng_required,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _exp1546_payload() -> dict[str, object]:
    return {
        "status": "complete",
        "milestone": "2026.04.118",
        "criteria_met": 13,
        "criteria_total": 14,
        "automata_contract_gate": {
            "adapter_ready": True,
            "automata_constraints_improved_contract_generation": True,
            "false_accept_rate": 0.0,
        },
        "satquest_verifier_gate": {
            "benchmark_ready": True,
            "solver_oracle_false_accepts": 3,
            "zero_solver_oracle_false_accepts": False,
            "carry_forward_to_119": "repair_oracle_false_accepts_before_acceptance_use",
        },
        "fr11_positive_utility_gate": {
            "continuous_self_learning_task": True,
            "no_model_weight_mutation": True,
            "soundness_mistakes": 0,
            "utility_delta": 0.0,
            "positive_utility_achieved": False,
            "status": "safety_only",
        },
        "arm_ebm_diagnostic_boundary": {
            "arm_ebm_diagnostic_ready": True,
            "logprob_available": False,
            "acceptance_authority": "deterministic_validators_only",
        },
        "thrml_next_scaling_gate": {
            "can_scale_further_in_software": True,
            "hardware_execution_claimed": False,
            "n256": {"ready": True, "simulator_only": True, "no_tsu_hardware_claim": True},
            "diverse_n64": {
                "ready": True,
                "simulator_only": True,
                "no_tsu_hardware_claim": True,
            },
        },
        "extropic_access_readiness_gate": {
            "readiness_packet_ready": True,
            "hardware_execution_claimed": False,
            "no_hardware_execution_claim": True,
        },
        "active_roadmap_modified": False,
        "conductor_modified": False,
        "honest_verdict": (
            "complete: milestone_118_13_of_14_criteria_met_satquest_fr11_limits"
        ),
    }


def _source_payloads() -> dict[str, dict[str, object]]:
    return {
        "exp1535": {
            "status": "complete",
            "contract_decoder_adapter_ready": True,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
        },
        "exp1536": {
            "status": "complete",
            "satquest_benchmark_ready": True,
            "solver_oracle_false_accepts": 3,
            "false_accept_rate": 0.166667,
        },
        "exp1538": {
            "status": "complete",
            "residual_drift_ledger_ready": True,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
        },
        "exp1539": {
            "status": "complete",
            "fr11_external_feedback_ready": True,
            "continuous_self_learning_task": True,
            "no_model_weight_mutation": True,
            "soundness_mistakes": 0,
            "utility_delta": 0.0,
            "positive_utility_promotion_ready": False,
        },
        "exp1540": {
            "status": "complete",
            "product_line_scale_ready": True,
            "branch_retired": False,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
        },
        "exp1541": {
            "status": "complete",
            "uncertainty_router_ready": True,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
        },
        "exp1542": {
            "status": "complete",
            "arm_ebm_diagnostic_ready": True,
            "deterministic_validators_final_authority": True,
            "logprob_available": False,
            "no_model_weight_mutation": True,
        },
        "exp1543": {
            "status": "complete",
            "thrml_parity_n256_schedule_ready": True,
            "parity_passed": True,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
        },
        "exp1544": {
            "status": "complete",
            "diverse_topology_parity_n64_ready": True,
            "parity_passed": True,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
        },
        "exp1545": {
            "status": "complete",
            "extropic_z1_readiness_packet_ready": True,
            "no_hardware_execution_claim": True,
            "research_roadmap_yaml_modified": False,
            "scripts_research_conductor_modified": False,
        },
    }


def _known_issues_with_rng_audit() -> str:
    return """
### NEW 2026-05-08 (10:40Z): THRML/Carnot Parity Independent-RNG Audit (.119+ MANDATORY)
Until this runs successfully, do not include the n=32-n=128 THRML parity
numbers in headline hardware-portability claims.
"""


def _retirement_context_text() -> str:
    return """
Legacy small-model headline claims remain blocked.
SATQuest acceptance before oracle repair remains blocked.
ARM/EBT soft-value acceptance authority remains blocked.
Extropic TSU/Z1/XTR-0 hardware execution claims remain blocked.
KV260 board claims remain blocked.
Model-weight mutation remains blocked.
Continuous self-learning is required for exp1555.
"""


def _research_complete_with_118() -> str:
    return """
- id: 2026.04.118
  title: Automata/SAT Runtime Contracts + Positive-Utility FR-11 + THRML Stress
"""


def test_scenario_report_060_activates_119_from_118_limits() -> None:
    """SCENARIO-REPORT-060: .119 activation exposes .118 carry-forward gates."""

    artifact, manifest = build_artifact(
        predecessor_retro=_exp1546_payload(),
        sources=_source_payloads(),
        missing_source_paths=[],
        conductor_log_text="| exp1546 | OK |",
        research_complete_text=_research_complete_with_118(),
        ops_status_text=_retirement_context_text(),
        ops_changelog_text=_retirement_context_text(),
        ops_known_issues_text=_known_issues_with_rng_audit(),
        roadmap_text="milestone: 2026.04.119\n",
        roadmap_next_text="",
        roadmap_doc_text=_retirement_context_text(),
        research_references_text=_retirement_context_text(),
        manifest_path="ops/milestone_119_activation_manifest.md",
        protected_files_unchanged=True,
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.119"
    assert artifact["predecessor_milestone"] == "2026.04.118"
    assert artifact["predecessor_criteria_met"] == 13
    assert artifact["predecessor_criteria_total"] == 14
    assert artifact["activation_manifest_complete"] is True
    assert artifact["prior_automata_ready"] is True
    assert artifact["prior_satquest_benchmark_ready"] is True
    assert artifact["prior_satquest_solver_oracle_false_accepts"] == 3
    assert artifact["prior_satquest_zero_solver_false_accepts"] is False
    assert artifact["prior_residual_drift_ready"] is True
    assert artifact["prior_fr11_safe_only"] is True
    assert artifact["prior_fr11_positive_utility"] is False
    assert artifact["prior_product_line_ready"] is True
    assert artifact["prior_claim_router_ready"] is True
    assert artifact["prior_arm_ebm_diagnostic_ready"] is True
    assert artifact["prior_thrml_n256_ready"] is True
    assert artifact["prior_thrml_diverse_n64_ready"] is True
    assert artifact["thrml_independent_rng_required"] is True
    assert artifact["prior_extropic_packet_ready"] is True
    assert artifact["research_complete_has_118_entry"] is True
    assert artifact["mandated_sota_models"] == MANDATED_SOTA_MODELS
    assert artifact["continuous_self_learning_required"] is True
    assert artifact["retired_headline_signals"] == RETIRED_HEADLINE_SIGNALS
    assert [track["track"] for track in artifact["allowed_119_tracks"]] == [
        track["track"] for track in ALLOWED_119_TRACKS
    ]
    assert artifact["same_roadmap_gate_fields"]["satquest_sota_reeval"] == {
        "prior_satquest_zero_solver_false_accepts": True
    }
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert "THRML independent-RNG audit" in manifest
    assert "SATQuest oracle repair" in manifest
    assert "Same-Roadmap Gates" in manifest
    assert "SATQuest acceptance before oracle repair" in manifest


def test_req_report_060_blocks_missing_and_unsafe_activation_inputs() -> None:
    """REQ-REPORT-060: missing and unsafe evidence stays explicit."""

    predecessor = _exp1546_payload()
    predecessor["criteria_met"] = 12
    predecessor["satquest_verifier_gate"] = {"solver_oracle_false_accepts": 0}
    predecessor["fr11_positive_utility_gate"] = {
        "no_model_weight_mutation": False,
        "soundness_mistakes": 0,
        "positive_utility_achieved": True,
    }
    sources = _source_payloads()
    sources["exp1535"]["contract_decoder_adapter_ready"] = False
    sources["exp1538"]["residual_drift_ledger_ready"] = False
    sources["exp1540"]["product_line_scale_ready"] = False
    sources["exp1541"]["uncertainty_router_ready"] = False
    sources["exp1542"]["arm_ebm_diagnostic_ready"] = False
    sources["exp1543"]["thrml_parity_n256_schedule_ready"] = False
    sources["exp1544"]["diverse_topology_parity_n64_ready"] = False
    sources["exp1545"]["extropic_z1_readiness_packet_ready"] = False

    artifact, manifest = build_artifact(
        predecessor_retro=predecessor,
        sources=sources,
        missing_source_paths=["results/missing_artifact.json"],
        conductor_log_text="",
        research_complete_text="- id: 2026.04.117\n",
        ops_status_text="",
        ops_changelog_text="",
        ops_known_issues_text="",
        roadmap_text="",
        roadmap_next_text="",
        roadmap_doc_text="",
        research_references_text="",
        manifest_path="ops/milestone_119_activation_manifest.md",
        protected_files_unchanged=False,
    )

    assert artifact["status"] == "blocked"
    assert artifact["activation_manifest_complete"] is False
    assert artifact["predecessor_criteria_met"] == 0
    assert artifact["predecessor_criteria_total"] == 0
    assert artifact["prior_automata_ready"] is False
    assert artifact["prior_satquest_zero_solver_false_accepts"] is True
    assert artifact["prior_fr11_safe_only"] is False
    assert artifact["prior_fr11_positive_utility"] is True
    assert artifact["thrml_independent_rng_required"] is False
    assert artifact["missing_source_paths"] == ["results/missing_artifact.json"]
    assert "predecessor .118 criteria are not 13 of 14" in artifact["blocked_reasons"]
    assert "listed source artifacts are missing" in artifact["blocked_reasons"]
    assert "protected files changed" in artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("passed:")
    assert "Manifest blocked" in manifest


def test_req_report_060_run_writes_bootstrap_manifest_and_terminal_json(tmp_path: Path) -> None:
    """REQ-REPORT-060: run writes bootstrap, markdown, and terminal artifact."""

    out_path = tmp_path / "results" / "experiment_1547_118_completion_archive_119_activation.json"
    manifest_path = tmp_path / "ops" / "milestone_119_activation_manifest.md"
    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    _write_json(
        tmp_path / "results" / "experiment_1546_milestone_118_retro.json", _exp1546_payload()
    )
    for exp_id, filename in SOURCE_FILES.items():
        _write_json(tmp_path / "results" / filename, _source_payloads()[exp_id])

    (tmp_path / "ops").mkdir(exist_ok=True)
    (tmp_path / "ops" / "conductor-log.md").write_text("| exp1546 | OK |\n", encoding="utf-8")
    (tmp_path / "ops" / "status.md").write_text(
        _retirement_context_text(), encoding="utf-8"
    )
    (tmp_path / "ops" / "changelog.md").write_text(
        _retirement_context_text(), encoding="utf-8"
    )
    (tmp_path / "ops" / "known-issues.md").write_text(
        _known_issues_with_rng_audit(), encoding="utf-8"
    )
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.04.119\n", encoding="utf-8")
    (tmp_path / "research-complete.yaml").write_text(
        _research_complete_with_118(), encoding="utf-8"
    )
    (tmp_path / "research-references.md").write_text(
        _retirement_context_text(), encoding="utf-8"
    )
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        _retirement_context_text(),
        encoding="utf-8",
    )

    artifact = run(
        root=tmp_path,
        out_path=out_path,
        manifest_path=manifest_path,
        protected_files_unchanged=True,
    )
    written = json.loads(out_path.read_text(encoding="utf-8"))
    manifest = manifest_path.read_text(encoding="utf-8")

    assert artifact == written
    assert written["status"] == "complete"
    assert written["manifest_path"] == "ops/milestone_119_activation_manifest.md"
    assert written["source_inputs_read"]["research-roadmap-next.yaml"]["exists"] is False
    assert "Allowed .119 Tracks" in manifest
    assert "FR-11 positive-utility-or-retire" in manifest


def test_req_report_060_defensive_helpers_stay_explicit(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-060: helpers keep missing and dirty inputs explicit."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _relative_path(tmp_path / "results" / "artifact.json") == "results/artifact.json"
    assert _relative_path(tmp_path / "loose.txt") == "loose.txt"
    assert _research_complete_has_118_entry("- id: 2026.04.118\n") is True
    assert _research_complete_has_118_entry('id: "2026.04.118"\n') is True
    assert _research_complete_has_118_entry("- id: 2026.04.117\n") is False
    assert _thrml_independent_rng_required(_known_issues_with_rng_audit()) is True
    assert _thrml_independent_rng_required("THRML parity without audit") is False
    assert _retirement_blocks_recorded(_retirement_context_text()) is True
    assert _retirement_blocks_recorded("legacy small-model only") is False

    loaded, missing = _load_sources(tmp_path / "results")
    assert loaded == {}
    assert missing == [f"results/{filename}" for filename in SOURCE_FILES.values()]

    class CleanResult:
        returncode = 0

    class DirtyResult:
        returncode = 1

    monkeypatch.setattr(activation119.subprocess, "run", lambda *args, **kwargs: CleanResult())
    assert _protected_files_clean(tmp_path) is True
    monkeypatch.setattr(activation119.subprocess, "run", lambda *args, **kwargs: DirtyResult())
    assert _protected_files_clean(tmp_path) is False

    def raise_os_error(*_args, **_kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(activation119.subprocess, "run", raise_os_error)
    assert _protected_files_clean(tmp_path) is True
