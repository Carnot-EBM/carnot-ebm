"""Tests for the Exp 2922 milestone .275 capstone artifact.

Spec refs: REQ-REPORT-2922, SCENARIO-REPORT-2922.

The capstone is a pure aggregation artifact. These tests build tiny upstream
JSON files and verify that the classifier keeps clean, flagged, blocked,
missing, pilot-only, and diagnostic-only evidence separate before deriving the
milestone-level paper, hardware, codegen, and FR-11 claim booleans.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v275_2922 as exp2922


def _write_json(root: Path, rel_path: str | Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _exp2909_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: archive_ready=true; archived_milestone=2026.05.274; activated_milestone=2026.05.275",
        "archive_ready": True,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "run_date": "20260523",
    }


def _exp2910_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: SOTA code-generation corrigendum executed",
        "codegen_corrigendum_ready": True,
        "candidate_generation_clean": True,
        "legacy_smoke_only": False,
        "aggregate_pass_at_1": 0.075,
        "aggregate_pass_at_k": 0.175,
        "pass_at_k_exceeds_pass_at_1": True,
        "flagged_adversarial": False,
        "corrigendum_pending": [],
        "inference_substrate": "live_llm_inference",
        "run_date": "20260523",
    }


def _exp2911_flagged_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: Exp 2910 code candidates labeled",
        "code_hallucination_verifier_ready": True,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        "inference_substrate": "deterministic_verifier",
        "run_date": "20260523",
    }


def _exp2912_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: same_basis_cpu_gibbs_baseline_ready_no_speedup_claim",
        "same_basis_cpu_baseline_ready": True,
        "speedup_claim_made": False,
        "inference_substrate": "cpu_sampler",
        "run_date": "20260523",
    }


def _exp2913_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: kv260_same_basis_hardware_cpu_speedup_claim_eligible",
        "kv260_claim_boundary_ready": True,
        "same_basis_verified": True,
        "hardware_speedup_claim_eligible": True,
        "speedup_claim_made": True,
        "speedup_ratio_median_by_sample_count": {"100": 19.13},
        "paper_claim_boundary": "Bounded n=64 KV260/CPU speedup claim only.",
        "matrix_row_candidate": {
            "eligible_for_matrix_v9": True,
            "eligible_for_paper_v6": True,
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "run_date": "20260523",
    }


def _exp2914_blocked_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "blocked_gatemate_toolchain_missing",
        "gatemate_toolchain_ready": False,
        "missing_toolchain": ["nextpnr-gatemate"],
        "no_flash_attempted": True,
        "inference_substrate": "hardware_preflight",
        "run_date": "20260523",
    }


def _exp2916_diagnostic_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: thrml_kv260_n64_simulator_parity_ready_no_hardware_claim",
        "thrml_kv260_parity_ready": True,
        "no_tsu_hardware_claim": True,
        "inference_substrate": "simulator_parity",
        "run_date": "20260523",
    }


def _exp2917_diagnostic_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: spilled_energy_micro_panel_diagnostic_ready",
        "spilled_energy_micro_panel_ready": True,
        "benchmark_claim_made": False,
        "claim_boundary": "diagnostic_only_no_benchmark_claim",
        "inference_substrate": "live_llm_inference",
        "run_date": "20260523",
    }


def _exp2918_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: verifier_process_rewards_updated_replay_scheduler",
        "online_self_learning_ready": True,
        "online_update_performed": True,
        "replay_scheduler_updated": True,
        "model_weights_mutated": False,
        "delta_overall": 0.85590778098,
        "delta_energy_proxy": 0.148821213249,
        "forgetting_rate": 0.0,
        "hardware_replay_used": True,
        "inference_substrate": "deterministic_verifier_plus_replay",
        "run_date": "20260523",
    }


def _exp2919_flagged_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: constraintbench mini direct optimization measured",
        "constraintbench_mini_ready": True,
        "feasibility_rate": 0.277778,
        "optimality_rate": 0.111111,
        "syntax_valid_rate": 0.277778,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        "inference_substrate": "live_llm_inference_plus_exact_verifier",
        "run_date": "20260523",
    }


def _exp2920_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: deterministic OpenComputer-style state verifier harness ready",
        "state_verifier_harness_ready": True,
        "llm_judge_used": False,
        "n_state_tasks": 4,
        "golden_state_pass_rate": 1.0,
        "negative_state_reject_rate": 1.0,
        "inference_substrate": "deterministic_state_verifier",
        "run_date": "20260523",
    }


def _exp2921_payload(*, flagged: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "honest_verdict": "complete: cross-corpus matrix v9 and paper-v6 claim boundary built",
        "cross_corpus_matrix_v9_built": True,
        "paper_claim_boundary_ready": True,
        "headline_eligible_rows": [
            "corpus:FoVer",
            "corpus:HaluEval_FEVER",
            "exp2910_sota_codegen",
            "exp2913_kv260_claim_boundary",
            "exp2918_fr11_process_rewards",
            "exp2920_state_verifier_harness",
        ],
        "clean_rows": [
            "corpus:FoVer",
            "corpus:HaluEval_FEVER",
            "exp2910_sota_codegen",
            "exp2913_kv260_claim_boundary",
            "exp2918_fr11_process_rewards",
            "exp2920_state_verifier_harness",
        ],
        "flagged_rows": ["exp2911_code_hallucination_verifier", "exp2919_constraintbench_mini"],
        "blocked_rows": ["exp2914_gatemate_toolchain"],
        "missing_rows": ["exp2915_gatemate_bitstream"],
        "diagnostic_only_rows": [
            "exp2916_thrml_parity",
            "exp2917_spilled_energy_micro_panel",
        ],
        "paper_v6_claim_boundary": {"ready": True},
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "run_date": "20260523",
    }
    if flagged:
        payload["flagged_adversarial"] = True
        payload["corrigendum_pending"] = [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}]
    return payload


def _write_scenario_sources(root: Path, *, matrix_flagged: bool = False) -> None:
    payloads = {
        exp2922.EXPECTED_ARTIFACTS["exp2909"].path: _exp2909_payload(),
        exp2922.EXPECTED_ARTIFACTS["exp2910"].path: _exp2910_payload(),
        exp2922.EXPECTED_ARTIFACTS["exp2911"].path: _exp2911_flagged_payload(),
        exp2922.EXPECTED_ARTIFACTS["exp2912"].path: _exp2912_payload(),
        exp2922.EXPECTED_ARTIFACTS["exp2913"].path: _exp2913_payload(),
        exp2922.EXPECTED_ARTIFACTS["exp2914"].path: _exp2914_blocked_payload(),
        exp2922.EXPECTED_ARTIFACTS["exp2916"].path: _exp2916_diagnostic_payload(),
        exp2922.EXPECTED_ARTIFACTS["exp2917"].path: _exp2917_diagnostic_payload(),
        exp2922.EXPECTED_ARTIFACTS["exp2918"].path: _exp2918_payload(),
        exp2922.EXPECTED_ARTIFACTS["exp2919"].path: _exp2919_flagged_payload(),
        exp2922.EXPECTED_ARTIFACTS["exp2920"].path: _exp2920_payload(),
        exp2922.EXPECTED_ARTIFACTS["exp2921"].path: _exp2921_payload(flagged=matrix_flagged),
    }
    for rel_path, payload in payloads.items():
        _write_json(root, rel_path, payload)


def test_req_report_2922_spec_is_declared() -> None:
    """REQ-REPORT-2922: OpenSpec declares the .275 capstone contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-2922" in spec
    assert "SCENARIO-REPORT-2922" in spec
    assert "hardware_speedup_claim_eligible" in spec


def test_scenario_report_2922_synthesizes_capstone_with_missing_gatemate(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2922: missing Exp 2915 is explicit, not inferred."""

    _write_scenario_sources(tmp_path)

    artifact = exp2922.build_artifact(tmp_path, started_s=5.0, now_s=7.0)

    required = {
        "honest_verdict",
        "paper_ready",
        "hardware_baselines_ready",
        "hardware_speedup_claim_eligible",
        "sota_code_row_repaired",
        "fr11_self_learning_clean",
        "clean_artifacts",
        "flagged_artifacts",
        "blocked_artifacts",
        "missing_artifacts",
        "pilot_only_artifacts",
        "diagnostic_only_artifacts",
        "headline_eligible_rows",
        "hardware_claim_boundary",
        "codegen_claim_boundary",
        "fr11_claim_boundary",
        "top_3_next_actions",
        "inference_substrate",
        "duration_s",
        "run_date",
    }
    assert required <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["paper_ready"] is True
    assert artifact["hardware_baselines_ready"] is True
    assert artifact["hardware_speedup_claim_eligible"] is True
    assert artifact["sota_code_row_repaired"] is True
    assert artifact["fr11_self_learning_clean"] is True
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert artifact["run_date"] == "20260523"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"

    assert artifact["clean_artifacts"] == [
        "exp2909",
        "exp2910",
        "exp2912",
        "exp2913",
        "exp2918",
        "exp2920",
        "exp2921",
    ]
    assert artifact["flagged_artifacts"] == ["exp2911", "exp2919"]
    assert artifact["blocked_artifacts"] == ["exp2914"]
    assert artifact["missing_artifacts"] == ["exp2915"]
    assert artifact["pilot_only_artifacts"] == []
    assert artifact["diagnostic_only_artifacts"] == ["exp2916", "exp2917"]
    assert artifact["headline_eligible_rows"] == _exp2921_payload()["headline_eligible_rows"]

    hardware = artifact["hardware_claim_boundary"]
    assert hardware["same_basis_cpu_baseline_ready"] is True
    assert hardware["kv260_claim_boundary_ready"] is True
    assert hardware["hardware_speedup_claim_eligible"] is True
    assert hardware["gatemate_toolchain_ready"] is False
    assert hardware["gatemate_bitstream_built"] is False
    assert hardware["thrml_kv260_parity_ready"] is True
    assert hardware["no_tsu_hardware_claim"] is True

    codegen = artifact["codegen_claim_boundary"]
    assert codegen["codegen_corrigendum_ready"] is True
    assert codegen["sota_code_row_repaired"] is True
    assert codegen["code_hallucination_verifier_ready"] is True
    assert any("DURATION_TOO_SHORT" in risk for risk in codegen["remaining_methodology_risks"])

    fr11 = artifact["fr11_claim_boundary"]
    assert fr11["online_self_learning_ready"] is True
    assert fr11["fr11_self_learning_clean"] is True
    assert fr11["delta_overall"] == pytest.approx(0.85590778098)
    assert fr11["forgetting_rate"] == pytest.approx(0.0)

    assert len(artifact["top_3_next_actions"]) == 3
    assert any("GateMate" in action for action in artifact["top_3_next_actions"])
    assert any("code hallucination" in action for action in artifact["top_3_next_actions"])
    assert any("ConstraintBench" in action for action in artifact["top_3_next_actions"])


def test_req_report_2922_matrix_flags_do_not_promote_or_hide_claims(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2922: a flagged matrix artifact stays flagged but source-clean headlines remain visible."""

    _write_scenario_sources(tmp_path, matrix_flagged=True)

    artifact = exp2922.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert "exp2921" in artifact["flagged_artifacts"]
    assert "exp2921" not in artifact["clean_artifacts"]
    assert artifact["paper_ready"] is True
    assert any("matrix v9" in action for action in artifact["top_3_next_actions"])


def test_req_report_2922_missing_cpu_baseline_blocks_hardware_speedup(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2922: Exp 2913 cannot underwrite speedup when Exp 2912 is missing."""

    _write_scenario_sources(tmp_path)
    (tmp_path / exp2922.EXPECTED_ARTIFACTS["exp2912"].path).unlink()

    artifact = exp2922.build_artifact(tmp_path, started_s=1.0, now_s=1.2)

    assert "exp2912" in artifact["missing_artifacts"]
    assert artifact["hardware_baselines_ready"] is False
    assert artifact["hardware_speedup_claim_eligible"] is False
    assert artifact["paper_ready"] is False
    assert artifact["hardware_claim_boundary"]["speedup_claim_made"] is False


def test_req_report_2922_write_artifact_persists_json(tmp_path: Path) -> None:
    """REQ-REPORT-2922: write_artifact persists the capstone JSON."""

    _write_scenario_sources(tmp_path)
    out = exp2922.write_artifact(tmp_path, started_s=0.0, now_s=0.25)

    assert out == tmp_path / exp2922.OUTPUT_REL_PATH
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == exp2922.SCHEMA
    assert payload["artifact"] == "experiment_2922_capstone_v275"
    assert payload["duration_s"] == pytest.approx(0.25)


def test_req_report_2922_defensive_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-2922: defensive parsing and classification keep bad inputs bounded."""

    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert exp2922.read_json(bad) == {}
    assert exp2922.read_json(tmp_path / "missing.json") == {}
    array = tmp_path / "array.json"
    array.write_text("[1]", encoding="utf-8")
    assert exp2922.read_json(array) == {}

    assert exp2922.classify_artifact("exp2915", {}, False) == "missing"
    assert (
        exp2922.classify_artifact(
            "exp2915",
            {
                "honest_verdict": "complete: gatemate_n16_ising_tile_bitstream_built",
                "gatemate_bitstream_built": True,
            },
            True,
        )
        == "clean"
    )
    assert (
        exp2922.classify_artifact(
            "exp2918",
            {
                **_exp2918_payload(),
                "online_update_performed": False,
                "replay_scheduler_updated": False,
            },
            True,
        )
        == "blocked"
    )
    assert (
        exp2922.classify_artifact(
            "exp2916",
            {**_exp2916_diagnostic_payload(), "no_tsu_hardware_claim": False},
            True,
        )
        == "blocked"
    )
    assert (
        exp2922.classify_artifact(
            "exp2909",
            {"honest_verdict": "partial: archive"},
            True,
        )
        == "blocked"
    )


def test_req_report_2922_extra_defensive_branches() -> None:
    """REQ-REPORT-2922: uncommon defensive branches stay covered and stable."""

    assert (
        exp2922.classify_artifact(
            "exp2909",
            {"honest_verdict": "complete: pilot", "pilot_only": True},
            True,
        )
        == "pilot_only"
    )
    assert exp2922._has_flags({"adversarial_verify_passed": False}) is True
    assert exp2922._has_flags({"adversarial_verify_flags": [{"kind": "X"}]}) is True
    assert exp2922._terminal_success(None) is False
    assert exp2922._paper_ready({}, [], {}) is False
    assert (
        exp2922._paper_ready(
            {
                "paper_claim_boundary_ready": True,
                "cross_corpus_matrix_v9_built": False,
            },
            [],
            {},
        )
        is False
    )
    assert exp2922._methodology_risks(
        {
            "expA": {"adversarial_verify_passed": False},
            "expB": {"flagged_adversarial": True, "corrigendum_pending": []},
        }
    ) == [
        "expA:adversarial_verify_passed:false",
        "expB:flagged_adversarial:true",
    ]
    assert len(exp2922._top_3_next_actions({})) == 3
