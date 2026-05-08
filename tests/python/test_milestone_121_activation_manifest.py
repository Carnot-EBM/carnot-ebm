"""Tests for the Exp 1574 `.121` activation manifest.

Spec: REQ-REPORT-063, SCENARIO-REPORT-063.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from carnot.reporting.milestone_121_activation_manifest import (
    ALLOWED_121_TRACKS,
    PRESERVED_CLAIM_BLOCKS,
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    _load_sources,
    _protected_files_clean,
    _read_json,
    _read_text,
    _relative_path,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _exp1572_payload() -> dict[str, object]:
    return {
        "status": "complete",
        "milestone": "2026.05.120",
        "next_milestone": "2026.05.121",
        "criteria_met": 10,
        "criteria_total": 14,
        "notable_successes": [
            "exp1564:thrml_vendoring",
            "exp1565:soft_gibbs_residual",
            "exp1566:candidate_warm_start",
            "exp1571:step_wise_ar_reinforce_baseline",
        ],
        "failures_or_partials": [
            {"experiment_id": "exp1561", "criterion": "kinetic_defense"},
            {"experiment_id": "exp1562", "criterion": "brain_linear_ar_rescue"},
            {"experiment_id": "exp1569", "status": "BLOCKED"},
            {"experiment_id": "exp1573", "status": "BLOCKED"},
        ],
        "carry_forward_gates_121": [
            {"source": "exp1569", "gate": "paper_v6_section_3_finalization"},
            {"source": "exp1565+exp1570", "gate": "soft_gibbs_residual_production_scale_n128"},
        ],
        "additional_carry_forwards_121": [
            {"source": "exp1573", "gate": "extropic_z1_readiness_packet_resumed_after_exp1573"},
            {"source": "exp1562", "gate": "brain_reinforce_training_dynamics_at_k15"},
            {"source": "exp1568", "gate": "fr11_v15_lambda_grpo_retention_reversal"},
        ],
        "terminal_verdicts": {
            "exp1569": "blocked_gate_check_failed",
            "exp1573": "blocked_gate_check_failed",
        },
        "honest_verdict": (
            "complete: milestone_120_10_of_14_criteria_met_paper_v6_exp1569"
            "_and_z1_exp1573_carried_to_121"
        ),
    }


def _source_payloads() -> dict[str, dict[str, object]]:
    return {
        "exp1561": {
            "status": "complete",
            "kinetic_defense_in_depth_validated": False,
            "honest_verdict": "complete_thrml_block_gibbs_falsifies_kinetic_security_parity",
        },
        "exp1562": {
            "status": "complete",
            "brain_linear_ar_rescue_validated": False,
            "phase_3_recommendation": "brain_dropped",
            "honest_verdict": "complete: falsified BRAIN+Linear-AR rescue widening",
        },
        "exp1564": {
            "status": "complete",
            "thrml_vendoring_complete": True,
            "candidate_warm_start_implemented": True,
            "no_tsu_hardware_claim": True,
        },
        "exp1565": {
            "status": "complete",
            "soft_gibbs_residual_implemented": True,
            "hard_brs_acceptance_rate": 0.0,
        },
        "exp1566": {
            "status": "complete",
            "candidate_warm_start_validated": True,
            "recommended_deployment_policy": "candidate_warm_start",
        },
        "exp1568": {
            "status": "complete",
            "mode_collapse_audit_complete": True,
            "reversal_recommended_count": 1,
            "retention_reversal_recommended_policy_ids": ["policy:residual_drift_repair:1552"],
        },
        "exp1569": {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "prior_failures field is missing or incomplete",
        },
        "exp1570": {
            "status": "complete",
            "jensen_bound_holds_for_all_beta": True,
            "optimal_beta_for_deployment": 0.1,
        },
        "exp1571": {
            "status": "complete",
            "step_wise_baseline_implemented": True,
            "gradient_variance_reduction_factor": 10.454576,
            "convergence_rate_matches_theorem_2": True,
        },
        "exp1573": {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "prior_failures field is missing or incomplete",
        },
    }


def _roadmap_context() -> str:
    return """
Phase-1 Software Ship Readiness Ledger
BRAIN REINFORCE Training Dynamics at k=15
OT verification framework adoption
DCCD + JSONSchemaBench Structured-Output Smoke on Mandated SOTA GGUFs
FR-11 v15 lambda-GRPO Patch + v14 Retention Reversal
Z1 drift correction
Tenstorrent Wormhole n150d Block-Gibbs Preflight
Microchip PolarFire SoC Adaptive K-PCD Prototype Preflight
Strix Point Secondary-Tier Rescope + KV260 Vivado-Lineage Retirement
"""


def _claim_blocks_context() -> str:
    return """
TSU/Z1 hardware execution claims remain blocked.
KV260 board claims without transcripts remain blocked.
Legacy small-model headline results remain blocked.
Soft energy/logprob scores as acceptance authority remain blocked.
"""


def test_scenario_report_063_activates_121_structured_gates() -> None:
    """SCENARIO-REPORT-063: .121 activation exposes downstream gate fields."""

    artifact, manifest = build_artifact(
        milestone_retro=_exp1572_payload(),
        sources=_source_payloads(),
        missing_source_paths=[],
        roadmap_text=_roadmap_context(),
        ops_known_issues_text=_claim_blocks_context(),
        architecture_text=_claim_blocks_context(),
        manifest_path="ops/milestone_121_activation_manifest.md",
        protected_files_unchanged=True,
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["activation_manifest_complete"] is True
    assert artifact["prior_failure_autofill_ready"] is True
    assert artifact["paper_v6_sampler_resume_ready"] is True
    assert artifact["extropic_packet_resume_ready"] is True
    assert artifact["brain_reinforce_training_ready"] is True
    assert artifact["ot_framework_adoption_ready"] is True
    assert artifact["dccd_jsonschema_smoke_ready"] is True
    assert artifact["fr11_v15_patch_ready"] is True
    assert artifact["phase1_ship_readiness_ready"] is True
    assert artifact["hardware_eval_ready"] is True
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["preserved_claim_blocks"] == PRESERVED_CLAIM_BLOCKS
    assert [row["track"] for row in artifact["allowed_121_tracks"]] == [
        row["track"] for row in ALLOWED_121_TRACKS
    ]
    assert "candidate warm-start" in " ".join(artifact["proved"])
    assert "BRAIN" in " ".join(artifact["falsified"])
    assert "exp1569" in " ".join(artifact["carried_forward"])
    assert "Phase-1 ship readiness" in manifest
    assert "DCCD/JSONSchemaBench SOTA smoke" in manifest
    assert "soft energy/logprob scores as acceptance authority" in manifest
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_report_063_blocks_missing_or_unsafe_inputs() -> None:
    """REQ-REPORT-063: missing evidence prevents terminal activation success."""

    bad_retro = _exp1572_payload()
    bad_retro["criteria_met"] = 9
    sources = _source_payloads()
    sources["exp1569"] = {"status": "complete"}
    sources["exp1571"] = {"status": "complete", "step_wise_baseline_implemented": False}

    artifact, manifest = build_artifact(
        milestone_retro=bad_retro,
        sources=sources,
        missing_source_paths=["results/missing.json"],
        roadmap_text="Phase-1 only",
        ops_known_issues_text="",
        architecture_text="",
        manifest_path="ops/milestone_121_activation_manifest.md",
        protected_files_unchanged=False,
    )

    assert artifact["status"] == "blocked"
    assert artifact["activation_manifest_complete"] is False
    assert artifact["prior_failure_autofill_ready"] is False
    assert artifact["paper_v6_sampler_resume_ready"] is False
    assert artifact["brain_reinforce_training_ready"] is False
    assert artifact["dccd_jsonschema_smoke_ready"] is False
    assert artifact["hardware_eval_ready"] is False
    assert "Exp 1572 does not report .120 completion" in artifact["blocked_reasons"]
    assert "listed source artifacts are missing" in artifact["blocked_reasons"]
    assert "protected files changed" in artifact["blocked_reasons"]
    assert "Manifest blocked" in manifest
    assert artifact["honest_verdict"].startswith("blocked:")


def test_req_report_063_run_writes_bootstrap_manifest_and_terminal_json(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-063: run writes in-progress, markdown, and terminal JSON."""

    out_path = tmp_path / "results" / "experiment_1574_120_completion_archive_121_activation.json"
    manifest_path = tmp_path / "ops" / "milestone_121_activation_manifest.md"
    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    _write_json(tmp_path / "results" / "experiment_1572_milestone_120_retro.json", _exp1572_payload())
    for exp_id, filename in SOURCE_FILES.items():
        _write_json(tmp_path / "results" / filename, _source_payloads()[exp_id])

    (tmp_path / "ops").mkdir(exist_ok=True)
    (tmp_path / "ops" / "known-issues.md").write_text(
        _claim_blocks_context() + _roadmap_context(), encoding="utf-8"
    )
    (tmp_path / "_bmad").mkdir(exist_ok=True)
    (tmp_path / "_bmad" / "architecture.md").write_text(
        _claim_blocks_context(), encoding="utf-8"
    )
    (tmp_path / "research-roadmap.yaml").write_text(
        _roadmap_context(), encoding="utf-8"
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "research_conductor.py").write_text(
        "# unchanged\n", encoding="utf-8"
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
    assert written["manifest_path"] == "ops/milestone_121_activation_manifest.md"
    assert written["source_inputs_read"]["results/experiment_1572_milestone_120_retro.json"]["exists"] is True
    assert "Allowed .121 Tracks" in manifest
    assert "Extropic Z1 readiness update" in manifest


def test_req_report_063_helpers_keep_missing_inputs_explicit(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-063: helper functions report missing inputs deterministically."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _relative_path(tmp_path / "results" / "artifact.json") == "results/artifact.json"
    assert _relative_path(tmp_path / "artifact.json") == "artifact.json"
    _write_json(tmp_path / "results" / SOURCE_FILES["exp1561"], _source_payloads()["exp1561"])
    loaded, missing = _load_sources(tmp_path / "results")
    assert loaded["exp1561"]["status"] == "complete"
    assert f"results/{SOURCE_FILES['exp1562']}" in missing
    monkeypatch.setattr(
        "carnot.reporting.milestone_121_activation_manifest.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    assert _protected_files_clean(tmp_path) is True
