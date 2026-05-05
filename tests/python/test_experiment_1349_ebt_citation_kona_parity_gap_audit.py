"""Tests for Exp 1349 EBT citation and Kona parity gap audit.

Spec refs: REQ-KONA-009, SCENARIO-KONA-009.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import ebt_citation_kona_parity_gap_audit as exp1349


def _thrml_artifact() -> dict[str, Any]:
    return {
        "status": "complete",
        "thrml_import_available": False,
        "energy_parity_max_abs_error": None,
        "sample_quality_proxy": None,
        "hardware_claim_allowed": False,
        "honest_verdict": "thrml_unavailable_mapping_notes_only_no_hardware_claim",
        "metadata": {
            "run_date": "20260505",
            "tsu_hardware_execution": False,
        },
    }


def _pbit_artifact() -> dict[str, Any]:
    return {
        "status": "complete",
        "hardware_claim_allowed": False,
        "kv260_claim_allowed": False,
        "honest_verdict": "cpu_only_update_dynamics_dual_bram_packet_ready_hardware_not_run",
        "metadata": {
            "run_date": "20260505",
            "synthesis_performed": False,
            "board_executed": False,
        },
        "reuse_factor_grid": [
            {"reuse_factor": 1, "cpu_kl_to_gibbs": 3.141314861474},
            {"reuse_factor": 2, "cpu_kl_to_gibbs": 0.04325904435},
            {"reuse_factor": 4, "cpu_kl_to_gibbs": 0.000412165565},
        ],
    }


def _token_health_artifact() -> dict[str, Any]:
    return {
        "status": "complete",
        "min_tokens_recovered": True,
        "empty_or_one_token_rate": 0.4,
        "entropy_production_rate_available": True,
        "topk_logprob_available": True,
        "headline_result_allowed": True,
        "honest_verdict": "token_health_recovered_certificate_prompt_multitoken",
        "models_used": [
            "unsloth/Qwen3.6-35B-A3B-GGUF",
            "unsloth/gemma-4-31B-it-GGUF",
        ],
    }


def _certificate_taxonomy_artifact() -> dict[str, Any]:
    return {
        "status": "complete",
        "parser_failure_count": 40,
        "semantic_failure_count": 30,
        "undergeneration_failure_count": 25,
        "unknown_state_mishandling_count": 4,
        "possible_hardcoded_solution_leakage_count": 35,
        "minimum_parseable_attempts_to_recover": 6,
        "source_metrics": {
            "exp1312_certificate_parse_rate": 0.71223,
            "exp1312_certificate_truthfulness_rate": 0.69697,
        },
        "honest_verdict": "diagnostic_complete_parse_gate_shortfall_parser_recovery_needed",
    }


def test_req_kona_009_builds_complete_claim_boundary_audit() -> None:
    """REQ-KONA-009: every required artifact field is populated conservatively."""
    artifact = exp1349.build_artifact(
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
        thrml_artifact=_thrml_artifact(),
        pbit_artifact=_pbit_artifact(),
        token_health_artifact=_token_health_artifact(),
        certificate_taxonomy_artifact=_certificate_taxonomy_artifact(),
    )

    exp1349.validate_artifact(artifact)
    assert exp1349.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["artifact_metadata"]["run_date"] == "20260505"
    assert artifact["external_dependency_claim_allowed"] is False
    assert artifact["honest_verdict"] == exp1349.HONEST_VERDICT
    assert len(artifact["ebt_citation_themes"]) >= 4
    assert len(artifact["kona_public_claims_mapped"]) >= 4
    assert len(artifact["parity_gaps"]) >= 5
    assert len(artifact["phase3_obligations"]) >= 5


def test_req_kona_009_ebt_themes_cover_required_research_neighborhood() -> None:
    """REQ-KONA-009: EBT themes include EBT, NRGPT, policy, optimizer, and code."""
    artifact = exp1349.build_artifact(
        thrml_artifact=_thrml_artifact(),
        pbit_artifact=_pbit_artifact(),
        token_health_artifact=_token_health_artifact(),
        certificate_taxonomy_artifact=_certificate_taxonomy_artifact(),
    )
    theme_names = {row["theme"] for row in artifact["ebt_citation_themes"]}
    theme_payload = json.dumps(artifact["ebt_citation_themes"], sort_keys=True)

    assert {
        "ebt_reasoning",
        "nrgpt_energy_recurrence",
        "ebt_policy_dynamic_compute",
        "intrinsic_optimizer_variants",
        "metacognitive_code_generation",
    } <= theme_names
    assert "code metacognition" in theme_payload
    assert "dynamic compute allocation" in theme_payload
    assert "ARM-to-EBM bridge" in theme_payload


def test_scenario_kona_009_maps_public_claims_to_local_gaps_without_overclaiming() -> None:
    """SCENARIO-KONA-009: Kona-style claims map to evidence and unsolved gaps."""
    artifact = exp1349.build_artifact(
        thrml_artifact=_thrml_artifact(),
        pbit_artifact=_pbit_artifact(),
        token_health_artifact=_token_health_artifact(),
        certificate_taxonomy_artifact=_certificate_taxonomy_artifact(),
    )
    mapped = {row["public_claim"]: row for row in artifact["kona_public_claims_mapped"]}

    assert mapped["non_autoregressive_ebm_reasoning_layer"]["claim_allowed_for_carnot"] is False
    assert mapped["valid_safe_permissible_state_scoring"]["claim_allowed_for_carnot"] is True
    assert mapped["hardware_portable_energy_execution"]["claim_allowed_for_carnot"] is False
    assert mapped["open_reproducible_local_certificates"]["claim_allowed_for_carnot"] is True
    assert "semantic certificate gate" in json.dumps(artifact["parity_gaps"])
    assert "THRML" in json.dumps(artifact["parity_gaps"])
    assert "Kona parity" in json.dumps(artifact["publication_claim_changes_needed"])


def test_req_kona_009_validation_rejects_external_claims_without_local_evidence() -> None:
    """REQ-KONA-009: external dependency claims stay false without reproducible evidence."""
    artifact = exp1349.build_artifact(
        thrml_artifact=_thrml_artifact(),
        pbit_artifact=_pbit_artifact(),
        token_health_artifact=_token_health_artifact(),
        certificate_taxonomy_artifact=_certificate_taxonomy_artifact(),
    )

    dishonest = dict(artifact)
    dishonest["external_dependency_claim_allowed"] = True
    with pytest.raises(ValueError, match="external_dependency_claim_allowed"):
        exp1349.validate_artifact(dishonest)

    missing = dict(artifact)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing"):
        exp1349.validate_artifact(missing)


def test_scenario_kona_009_run_experiment_writes_in_progress_then_complete(
    tmp_path: Path,
) -> None:
    """SCENARIO-KONA-009: runner persists a terminal audit from local artifacts only."""
    results = tmp_path / "results"
    results.mkdir()
    source_paths = {
        "thrml_path": results / "experiment_1347.json",
        "pbit_path": results / "experiment_1348.json",
        "token_health_path": results / "experiment_1323.json",
        "certificate_taxonomy_path": results / "experiment_1324.json",
    }
    source_paths["thrml_path"].write_text(json.dumps(_thrml_artifact()), encoding="utf-8")
    source_paths["pbit_path"].write_text(json.dumps(_pbit_artifact()), encoding="utf-8")
    source_paths["token_health_path"].write_text(
        json.dumps(_token_health_artifact()), encoding="utf-8"
    )
    source_paths["certificate_taxonomy_path"].write_text(
        json.dumps(_certificate_taxonomy_artifact()), encoding="utf-8"
    )
    output_path = results / "experiment_1349_ebt_citation_kona_parity_gap_audit.json"

    writes: list[str] = []
    artifact = exp1349.run_experiment(
        project_root=tmp_path,
        run_date="20260505",
        output_path=output_path,
        write_observer=lambda _path, payload: writes.append(str(payload["status"])),
        **source_paths,
    )

    assert writes == ["in_progress", "complete"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["external_dependency_claim_allowed"] is False
