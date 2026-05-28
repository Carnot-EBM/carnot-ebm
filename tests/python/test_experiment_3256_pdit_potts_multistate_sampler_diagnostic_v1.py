"""Tests for Exp 3256 p-dit/Potts multi-state sampler diagnostic.

Spec refs: REQ-POTTS-009, SCENARIO-POTTS-009.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.analysis import pdit_potts_multistate_sampler_diagnostic_v1 as mod


REQUIRED_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "principle_annotations",
    "pdit_potts_mapping_ready",
    "candidate_verifier_row_types",
    "q_state_energy_mapping",
    "exact_fallback_preserved",
    "hardware_speedup_claim_allowed",
    "retired_pimi_scope_reopened",
    "thrml_scaling_sweep_reopened",
    "future_gated_experiment_contract",
    "random_seed",
    "reproducibility_checksum",
    "honest_verdict",
}


def test_req_potts_009_spec_anchor_exists() -> None:
    """REQ-POTTS-009: OpenSpec declares the Exp 3256 diagnostic manifest."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/potts-sampler/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-POTTS-009" in spec
    assert "SCENARIO-POTTS-009" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "hardware_speedup_claim_allowed=false" in spec
    assert "exact fallback" in spec


def test_req_potts_009_candidate_rows_use_q_state_partial_credit() -> None:
    """REQ-POTTS-009-2: candidate verifier rows expose q-state labels and energies."""

    rows = mod.build_candidate_verifier_row_types()
    mappings = mod.build_q_state_energy_mapping(rows)

    assert {row["row_type"] for row in rows} == {
        "opencomputer_state_verifier_partial_credit",
        "logitext_partial_smt_context_row",
    }
    assert all(row["q"] > 2 for row in rows)
    assert all("partial" in " ".join(row["state_labels"]) for row in rows)
    assert {mapping["row_type"] for mapping in mappings} == {row["row_type"] for row in rows}

    for mapping in mappings:
        energies = list(mapping["energy_table"].values())
        assert energies == sorted(energies)
        assert len(energies) == mapping["q"]
        assert mapping["binary_one_hot_spin_count"] == mapping["q"]
        assert mapping["invalid_binary_one_hot_state_count"] == (2 ** mapping["q"]) - mapping["q"]
        assert mapping["potts_variable"]["state_count"] == mapping["q"]
        assert mapping["pdit_variable"]["alphabet_size"] == mapping["q"]
        assert mapping["exact_fallback_check"]["preserved"] is True


def test_scenario_potts_009_builds_complete_manifest_without_reopened_scope() -> None:
    """SCENARIO-POTTS-009: build the diagnostic manifest with exact fallback gates."""

    artifact = mod.build_artifact()

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["experiment_id"] == "exp3256"
    assert artifact["task_id"] == "exp3256-pdit-potts-multistate-sampler-diagnostic-v1"
    assert artifact["milestone"] == "2026.05.301"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["pdit_potts_mapping_ready"] is True
    assert artifact["exact_fallback_preserved"] is True
    assert artifact["hardware_speedup_claim_allowed"] is False
    assert artifact["retired_pimi_scope_reopened"] is False
    assert artifact["thrml_scaling_sweep_reopened"] is False
    assert artifact["random_seed"] == 3256
    assert artifact["honest_verdict"].startswith("complete:")
    assert all(
        denied not in artifact["honest_verdict"].lower()
        for denied in ("live hardware", "thrml", "kona", "speedup")
    )
    assert mod.checksum_for(artifact) == artifact["reproducibility_checksum"]

    contract = artifact["future_gated_experiment_contract"]
    assert contract["contract_allowed"] is True
    assert {gate["gate"]: gate["required"] for gate in contract["preconditions"]} == {
        "exact_fallback_preserved": True,
        "hardware_speedup_claim_allowed": False,
        "retired_pimi_scope_reopened": False,
        "thrml_scaling_sweep_reopened": False,
    }
    assert all(claim["allowed"] is False for claim in contract["blocked_claims"])


def test_req_potts_009_validation_rejects_missing_or_dishonest_fields() -> None:
    """REQ-POTTS-009-1/3/4: validation fails closed on schema and claim errors."""

    artifact = mod.build_artifact()
    mod.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("q_state_energy_mapping")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    dishonest_speedup = dict(artifact)
    dishonest_speedup["hardware_speedup_claim_allowed"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim_allowed"):
        mod.validate_artifact(dishonest_speedup)

    reopened_pimi = dict(artifact)
    reopened_pimi["retired_pimi_scope_reopened"] = True
    with pytest.raises(ValueError, match="retired_pimi_scope_reopened"):
        mod.validate_artifact(reopened_pimi)

    reopened_thrml = dict(artifact)
    reopened_thrml["thrml_scaling_sweep_reopened"] = True
    with pytest.raises(ValueError, match="thrml_scaling_sweep_reopened"):
        mod.validate_artifact(reopened_thrml)

    no_fallback = dict(artifact)
    no_fallback["exact_fallback_preserved"] = False
    with pytest.raises(ValueError, match="exact_fallback_preserved"):
        mod.validate_artifact(no_fallback)

    q2_mapping = json.loads(json.dumps(artifact))
    q2_mapping["q_state_energy_mapping"][0]["q"] = 2
    with pytest.raises(ValueError, match="q > 2"):
        mod.validate_artifact(q2_mapping)

    row_fallback_removed = json.loads(json.dumps(artifact))
    row_fallback_removed["q_state_energy_mapping"][0]["exact_fallback_check"]["preserved"] = False
    with pytest.raises(ValueError, match="preserve exact fallback"):
        mod.validate_artifact(row_fallback_removed)

    bad_verdict = dict(artifact)
    bad_verdict["honest_verdict"] = "complete: THRML speedup evidence"
    bad_verdict["reproducibility_checksum"] = mod.checksum_for(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_scenario_potts_009_writer_persists_stable_json(tmp_path: Path) -> None:
    """SCENARIO-POTTS-009: writer persists deterministic, validated JSON."""

    out_path = tmp_path / "experiment_3256_pdit_potts_multistate_sampler_diagnostic_v1.json"

    written = mod.write_artifact(out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    second = mod.write_artifact(out_path)

    assert written == payload
    assert second == payload
    assert payload["reproducibility_checksum"] == mod.checksum_for(payload)
    assert payload["candidate_verifier_row_types"][0]["why_q_state_is_natural"]
    assert payload["q_state_energy_mapping"][0]["exact_fallback_check"]["authority"]
