"""Tests for the Exp 1576 paper-v6 Section 3 sampler draft.

Spec: REQ-PUBLISH-022, SCENARIO-PUBLISH-024.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_1576_paper_v6_section_3_sampler_draft_resumed.json"
)
DRAFT_PATH = REPO_ROOT / "docs" / "research-notes" / "paper-v6-section-3-sampler-draft.md"
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "publication" / "spec.md"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "draft_path",
    "paper_v6_sampler_section_draft_ready",
    "kinetic_security_caveat_included",
    "brain_training_dynamics_open_question_included",
    "no_hardware_execution_claim",
    "honest_verdict",
}

REQUIRED_DRAFT_HEADINGS = [
    "## THRML vendored sampler",
    "## Candidate warm-start",
    "## Soft-Gibbs Residual",
    "## Kinetic-security caveat",
    "## SpecAnn rejection",
    "## BRAIN expressivity vs training-dynamics open question",
]

REQUIRED_EVIDENCE_PATHS = [
    "results/experiment_1561_kinetic_defense_zero_coupling_test.json",
    "results/experiment_1562_brain_linear_ar_k_sweep_extended.json",
    "results/experiment_1563_specann_rejection_architecture_record.json",
    "results/experiment_1564_thrml_vendored_block_gibbs_replacement.json",
    "results/experiment_1565_soft_gibbs_residual_implementation.json",
    "results/experiment_1566_candidate_warm_start_vs_cold_start_benchmark.json",
    "results/experiment_1570_soft_gibbs_coverage_bound_empirical_verification.json",
    "results/experiment_1571_step_wise_baseline_AR_REINFORCE.json",
]


def _draft_text() -> str:
    return DRAFT_PATH.read_text(encoding="utf-8")


def _artifact() -> dict[str, object]:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_req_publish_022_spec_anchor_exists() -> None:
    """REQ-PUBLISH-022, SCENARIO-PUBLISH-024: Exp 1576 is spec-anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PUBLISH-022" in spec
    assert "SCENARIO-PUBLISH-024" in spec
    assert "experiment_1576_paper_v6_section_3_sampler_draft_resumed.json" in spec
    assert "paper-v6-section-3-sampler-draft.md" in spec


def test_req_publish_022_artifact_schema_and_terminal_flags() -> None:
    """REQ-PUBLISH-022: the deliverable records the required honest boundaries."""

    artifact = _artifact()

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["draft_path"] == "docs/research-notes/paper-v6-section-3-sampler-draft.md"
    assert artifact["paper_v6_sampler_section_draft_ready"] is True
    assert artifact["kinetic_security_caveat_included"] is True
    assert artifact["brain_training_dynamics_open_question_included"] is True
    assert artifact["no_hardware_execution_claim"] is True
    assert artifact["honest_verdict"] == (
        "paper_v6_section_3_sampler_draft_ready_for_exp1579_ot_integration"
    )


def test_scenario_publish_024_draft_has_required_sections_and_evidence() -> None:
    """SCENARIO-PUBLISH-024: draft covers the .120 sampler evidence explicitly."""

    text = _draft_text()

    for heading in REQUIRED_DRAFT_HEADINGS:
        assert heading in text
    for evidence_path in REQUIRED_EVIDENCE_PATHS:
        assert (REPO_ROOT / evidence_path).exists()
        assert evidence_path in text

    assert "THRML 0.1.3" in text
    assert "Apache-2.0" in text
    assert "candidate_warm_start_validated" in text
    assert "hard_brs_acceptance_rate = 0.0" in text
    assert "optimal_beta_for_deployment = 0.1" in text
    assert "phase_3_recommendation = brain_dropped" in text
    assert "step-wise baseline" in text


def test_scenario_publish_024_claim_boundaries_are_explicit() -> None:
    """SCENARIO-PUBLISH-024: negative findings are not rewritten as wins."""

    text = _draft_text()
    lower = text.lower()

    assert "does not claim thrml security parity" in lower
    assert "thrml_security_parity_with_single_site_gibbs = false" in text
    assert "kinetic defense-in-depth remains an unresolved sampler-security question" in lower
    assert "does not claim extropic hardware execution" in lower
    assert "simulator_only = true" in text
    assert "no_tsu_hardware_claim = true" in text
    assert "brain-as-published is rejected for phase 3" in lower
    assert "training-dynamics question remains open" in lower
    assert "specann remains rejected" in lower

    forbidden_positive_claims = [
        "thrml provides security parity",
        "thrml proves security parity",
        "ran on extropic hardware",
        "executed on extropic hardware",
        "z1 hardware execution",
        "brain is validated for phase 3",
    ]
    for phrase in forbidden_positive_claims:
        assert phrase not in lower


def test_scenario_publish_024_integration_checklist_targets_active_main_tex() -> None:
    """SCENARIO-PUBLISH-024: checklist gives insertion points, not a rewrite."""

    text = _draft_text()

    assert "## Paper-v6 integration checklist" in text
    assert "docs/papers/paper-v6/main.tex is not present" in text
    assert "docs/arxiv-paper/main.tex:756" in text
    assert "docs/arxiv-paper/main.tex:777" in text
    assert "docs/arxiv-paper/main.tex:804" in text
    assert r"\section{Hardware Acceleration \& Sampling Limits}" in text
    assert r"\subsection{The detailed-balance audit (exp1094)}" in text
    assert r"\subsection{Same-basis CPU-vs-FPGA timing remains open}" in text
    assert "Do not perform a wholesale rewrite" in text
    assert "Ready for exp1579 OT framework integration: yes" in text
