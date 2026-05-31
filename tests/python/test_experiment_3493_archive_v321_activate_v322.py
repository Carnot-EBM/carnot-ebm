"""Tests for exp3493 — Archive v321 and Activate v322.

Covers REQ-RETRO-010 (SCENARIO-RETRO-010): milestone archive artifacts must
capture the key finding (infra-loss vs science refutation), carry the
correct publication gate status, and emit a `complete:` prefixed
honest_verdict.

All logic is exercised by importing the script module and calling the
internal functions directly — no subprocess, no filesystem side-effects
from the functions under test.
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.experiment_3493_archive_v321_activate_v322 as module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_UPSTREAM = {
    "exp3486": {
        "honest_verdict": "complete: minimal_sufficient_entropy_beta_found",
        "minimal_sufficient_beta": 0.1,
        "minimal_beta_depends_on_grounding": True,
        "recommended_phase5_default": "entropy_beta=0.100",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    },
    "exp3488": {
        "honest_verdict": "complete: fover_g2_package_regression_clean_external_ask_ready",
        "package_reproduced_auroc": 0.9131,
        "package_auroc_within_ci": True,
        "g2_met": False,
        "external_run_pending": True,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    },
    "exp3489": {
        "honest_verdict": "complete: blocked_kv260_ssh_unreachable",
        "kv260_terminal_state_reached": False,
        "inference_substrate": "hardware_smoke",
    },
    "exp3492": {
        "honest_verdict": "blocked_gate_check_failed",
        "status": "blocked",
    },
}


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-010: artifact structure + honest verdict prefix
# ---------------------------------------------------------------------------


def test_retro_honest_verdict_starts_with_complete():
    """REQ-RETRO-010 SCENARIO-RETRO-010: honest_verdict must start with 'complete:'."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["honest_verdict"].startswith("complete:"), (
        f"honest_verdict must start with 'complete:', got: {artifact['honest_verdict']!r}"
    )


def test_retro_archive_ready_flag():
    """REQ-RETRO-010: archive_v321_activate_v322_ready must be True."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["archive_v321_activate_v322_ready"] is True


def test_retro_p01_infra_loss_not_science():
    """REQ-RETRO-010: p01_status must indicate infra-loss, not science refutation."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["p01_status"] == "open_infra_loss_not_science"
    assert artifact["p01_science_unresolved"] is True
    assert artifact["p01_not_refuted"] is True


def test_retro_p01_consecutive_infra_losses():
    """REQ-RETRO-010: three consecutive infra losses must be recorded."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["p01_consecutive_infra_losses"] == 3


def test_retro_g2_status_propagated():
    """REQ-RETRO-010: G2 status from exp3488 must propagate correctly."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    gate_status = artifact["publication_gate_status"]
    assert gate_status["G2_independent_reproducer"] is False
    assert gate_status["G2_external_run_pending"] is True
    assert gate_status["G2_package_auroc"] == pytest.approx(0.9131)
    assert gate_status["paper_ready"] is False
    assert gate_status["sole_unmet_gate"] == "G2"


def test_retro_g1_g3_g4_met():
    """REQ-RETRO-010: G1, G3, G4 must be True (only G2 unmet)."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    gate_status = artifact["publication_gate_status"]
    assert gate_status["G1_headline_measured"] is True
    assert gate_status["G3_prose_narrowing_clean"] is True
    assert gate_status["G4_numbers_trace_to_artifacts"] is True


def test_retro_fr11_beta_propagated():
    """REQ-RETRO-010: FR-11 Phase-5 entropy beta must propagate from exp3486."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["fr11_phase5_entropy_beta_default"] == pytest.approx(0.1)
    assert artifact["fr11_phase5_grounding_dependence_confirmed"] is True


def test_retro_schema_field():
    """REQ-RETRO-010: schema must be carnot.operational_retro.v65."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["schema"] == "carnot.operational_retro.v65"


def test_retro_milestone_fields():
    """REQ-RETRO-010: milestone_archived and milestone_activated must be set."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["milestone_archived"] == "2026.05.321"
    assert artifact["milestone_activated"] == "2026.05.322"


def test_retro_inference_substrate():
    """REQ-RETRO-010: inference_substrate must be aggregation_from_upstream_artifacts."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"


def test_retro_duration_positive():
    """REQ-RETRO-010: duration_s must be positive (floored at 0.001)."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic() - 0.5)
    assert artifact["duration_s"] > 0


def test_retro_reproducibility_checksum_present():
    """REQ-RETRO-010: reproducibility_checksum must be a non-empty hex string."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert isinstance(artifact["reproducibility_checksum"], str)
    assert len(artifact["reproducibility_checksum"]) > 0


def test_retro_forward_gap_non_empty():
    """REQ-RETRO-010: forward_gap_top must name the two-path P0.1 routing approach."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert "CPU Sudoku" in artifact["forward_gap_top"]
    assert "cached-corpus" in artifact["forward_gap_top"]
    assert "sonnet" in artifact["forward_gap_top"]


def test_retro_cited_upstream_artifacts():
    """REQ-RETRO-010: cited_upstream_artifacts must list all four upstream sources."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    ids = {a["experiment_id"] for a in artifact["cited_upstream_artifacts"]}
    assert ids == {"exp3486", "exp3488", "exp3489", "exp3492"}


def test_retro_field_provenance_keys():
    """REQ-RETRO-010: field_provenance must annotate key fields."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    provenance = artifact["field_provenance"]
    required_keys = {
        "honest_verdict",
        "inference_substrate",
        "archive_v321_activate_v322_ready",
        "publication_gate_status",
        "duration_s",
        "reproducibility_checksum",
        "cited_upstream_artifacts",
    }
    for key in required_keys:
        assert key in provenance, f"field_provenance missing key: {key!r}"


def test_retro_experiments_completed_count():
    """REQ-RETRO-010: experiments_completed must include at least 8 entries for .321."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert len(artifact["experiments_completed"]) >= 8


def test_retro_v322_infra_fix_flag():
    """REQ-RETRO-010: v322_infra_fix_not_science must be True to distinguish fix type."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["v322_infra_fix_not_science"] is True


def test_retro_json_serializable():
    """REQ-RETRO-010: the full artifact must be JSON-serializable."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    serialized = json.dumps(artifact)  # must not raise
    roundtripped = json.loads(serialized)
    assert roundtripped["honest_verdict"].startswith("complete:")
