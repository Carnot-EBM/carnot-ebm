"""
Tests for exp1095 Phase 3a DBAE-EBM pre-prototype threat model.

These tests verify the THREAT DETECTION METHODS, not the DBAE-EBM prototype itself.
The prototype does not exist yet — this is the adversarial review round that precedes it.

REQ-PHASE3A-001: threat model document must be written and parseable before prototype code
REQ-PHASE3A-002: all 5 attack patterns must be documented
REQ-PHASE3A-003: instrumentation checklist (D-01 through D-10) must be complete
"""

import json
import os
import re

import pytest


THREAT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../docs/research-notes/phase3a-dbae-ebm-threat-model.md",
)
RESULT_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../results/experiment_1095_phase3a_dbae_ebm_adversarial_round.json",
)


def _read_threat_model():
    with open(THREAT_MODEL_PATH, encoding="utf-8") as f:
        return f.read()


# REQ-PHASE3A-001
def test_threat_model_document_written_and_parseable():
    """Threat model document exists, is non-empty, and has the expected title."""
    assert os.path.exists(THREAT_MODEL_PATH), (
        f"Threat model not found at {THREAT_MODEL_PATH}. "
        "exp1095 must produce this file before prototype code is written."
    )
    content = _read_threat_model()
    assert len(content) > 2000, (
        "Threat model is suspiciously short — expected detailed analysis of 5 attack patterns."
    )
    assert "Phase 3a DBAE-EBM Pre-Prototype Threat Model" in content, (
        "Threat model must have the canonical title for exp1095."
    )
    # Must have the architecture description box
    assert "[-1,1]^d" in content or "[-1, 1]^d" in content, (
        "Threat model must describe the DBAE bounded latent space."
    )


# REQ-PHASE3A-002
def test_all_5_attack_patterns_documented():
    """All 5 required attack patterns appear in the threat model with full structure."""
    content = _read_threat_model()
    expected_patterns = [
        "Degenerate Identity Encoder",
        "Decoder LM-Prior",
        "EBM Converging to Constants",
        "Verifier Joint Null-Space",
        "Bottleneck Collapse",
    ]
    for pattern in expected_patterns:
        assert pattern in content, (
            f"Attack pattern '{pattern}' not found in threat model. "
            "All 5 pre-prototype attack patterns must be documented."
        )

    # Each pattern must have all four required subsections
    required_subsections = [
        "How prototype can fail silently",
        "Why acceptance gate misses it",
        "Required instrumentation",
        "Minimum detection test",
    ]
    for subsection in required_subsections:
        count = content.count(subsection)
        assert count >= 5, (
            f"Subsection '{subsection}' appears {count} times — expected ≥5 "
            "(once per attack pattern). Every attack must document all four aspects."
        )


# REQ-PHASE3A-003
def test_instrumentation_checklist_complete():
    """The threat model contains a complete instrumentation checklist (D-01 through D-10)."""
    content = _read_threat_model()

    # All 10 diagnostics must appear
    for i in range(1, 11):
        tag = f"D-{i:02d}"
        assert tag in content, (
            f"Diagnostic {tag} is missing from the instrumentation checklist. "
            "All 10 diagnostics must be wired before any DBAE-EBM training run."
        )

    # Cross-phase dependency section must be present
    assert "Cross-Phase Dependency" in content, (
        "Threat model must analyse cross-phase dependency (Phase 1c pre-condition from exp1093)."
    )

    # Decentralization risk must be assessed
    assert "Decentralization Risk" in content or "decentralization" in content.lower(), (
        "Threat model must assess decentralization risk per CLAUDE.md Rule 2."
    )

    # Hardware portability must be assessed
    assert "Hardware Portability" in content or "hardware portability" in content.lower(), (
        "Threat model must assess hardware portability per CLAUDE.md Rule 5."
    )

    # Pre-condition from exp1093 must be referenced
    assert "exp1093" in content, (
        "Threat model must reference exp1093 result to verify Phase 1c pre-condition."
    )

    # Acceptance gates per stage must be present
    assert "Acceptance Gate" in content or "acceptance gate" in content.lower(), (
        "Threat model must include per-stage acceptance gates."
    )


def test_result_artifact_schema():
    """exp1095 result artifact exists and has all required fields with correct types."""
    assert os.path.exists(RESULT_PATH), (
        f"Result artifact not found at {RESULT_PATH}. "
        "Run scripts/experiment_1095_phase3a_dbae_ebm_adversarial_round.py to produce it."
    )
    with open(RESULT_PATH, encoding="utf-8") as f:
        artifact = json.load(f)

    required_fields = {
        "experiment": int,
        "threat_model_written": bool,
        "threat_model_path": str,
        "attack_patterns_documented": int,
        "instrumentation_checklist_complete": bool,
        "decentralization_risk_assessed": bool,
        "hardware_portability_assessed": bool,
        "pre_conditions_verified": bool,
        "tests_passing": int,
        "honest_verdict": str,
    }
    for field, expected_type in required_fields.items():
        assert field in artifact, f"Required field '{field}' missing from artifact."
        assert isinstance(artifact[field], expected_type), (
            f"Field '{field}' has type {type(artifact[field]).__name__}, "
            f"expected {expected_type.__name__}."
        )

    assert artifact["experiment"] == 1095
    assert artifact["attack_patterns_documented"] == 5
    assert artifact["honest_verdict"] in (
        "threat_model_complete",
        "threat_model_partial",
        "failed",
    )
    assert artifact["threat_model_path"] == "docs/research-notes/phase3a-dbae-ebm-threat-model.md"
