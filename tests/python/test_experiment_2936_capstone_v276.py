"""Tests for the Exp 2936 milestone .276 capstone.

Spec refs: REQ-REPORT-2936, SCENARIO-REPORT-2936.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v276_2936 as exp2936


REQUIRED_FIELDS = {
    "honest_verdict",
    "milestone",
    "paper_ready",
    "hardware_speedup_claim_eligible",
    "gate_mate_speedup_claim_eligible",
    "evidence_boundary_repaired",
    "sota_structured_generation_clean",
    "fr11_self_learning_clean",
    "clean_artifacts",
    "flagged_artifacts",
    "blocked_artifacts",
    "missing_artifacts",
    "projection_only_artifacts",
    "diagnostic_only_artifacts",
    "pilot_only_artifacts",
    "row_classification_counts",
    "top_three_next_actions",
    "source_artifact_checksums",
    "no_new_llm_call",
    "no_new_hardware_run",
    "inference_substrate",
    "duration_s",
    "run_date",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exp2923_payload() -> dict[str, Any]:
    return {
        "honest_verdict": (
            "complete: archive_ready=true; archived_milestone=2026.05.275; "
            "activated_milestone=2026.05.276"
        ),
        "archive_ready": True,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "run_date": "20260523",
    }


def _exp2924_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: aggregation metadata corrigendum written",
        "aggregation_metadata_clean": True,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        "adversarial_audit_rerun": {"flagged": False, "findings": []},
        "upstream_flagged_rows_preserved": [{"identifier": "exp2911"}],
        "metadata_false_positive_findings": [{"kind": "DURATION_TOO_SHORT"}],
        "no_new_llm_call": True,
        "no_new_hardware_run": True,
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _exp2925_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: taxonomy provenance corrigendum clean",
        "taxonomy_corrigendum_clean": True,
        "code_hallucination_verifier_ready": True,
        "deterministic_verifier_no_new_llm_call": True,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        "adversarial_audit_rerun": {"flagged": False, "findings": []},
        "syntax_error_rate": 0.38125,
        "undefined_name_rate": 0.03125,
        "true_test_failure_rate": 0.009375,
        "no_new_llm_call": True,
        "no_new_hardware_run": True,
        "inference_substrate": "deterministic_verifier",
    }


def _exp2926_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: constraintbench constrained-output rerun measured",
        "constraintbench_corrigendum_ready": True,
        "flagged_adversarial": False,
        "syntax_valid_rate": 0.333333,
        "feasibility_rate_overall": 0.3,
        "optimality_rate_given_feasible": 0.444444,
        "n_tasks": 30,
        "duration_s": 68.1,
        "reproducibility_checksum": "abc",
        "inference_substrate": "live_llm_inference",
    }


def _exp2927_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "blocked_constraints_missing",
        "gatemate_himbaechel_ready": True,
        "nextpnr_device_supported": True,
        "constraints_ready": False,
        "constraint_paths_present": [],
        "missing_toolchain": [],
        "tool_paths": {
            "nextpnr-himbaechel": "/opt/oss-cad-suite/bin/nextpnr-himbaechel",
            "gmpack": "/opt/oss-cad-suite/bin/gmpack",
        },
        "inference_substrate": "hardware_toolchain_preflight",
    }


def _exp2929_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "blocked_gatemate_bitstream_missing",
        "gatemate_flash_smoke_ready": False,
        "blocker": "missing exp2928 artifact",
        "flash_attempted": False,
        "speedup_claim_allowed": False,
        "inference_substrate": "physical_board_smoke",
    }


def _exp2930_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: kv260 projection ready",
        "kv260_scaling_projection_ready": True,
        "projection_only": True,
        "not_a_speedup_claim": True,
        "no_new_hardware_run": True,
        "inference_substrate": "aggregation_plus_simulation",
    }


def _exp2931_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "blocked_z3_execution_incomplete",
        "logic_verifier_mini_ready": False,
        "parseability_rate": 0.0,
        "z3_execution_rate": 0.0,
        "answer_accuracy": 0.0,
        "inference_substrate": "live_llm_inference_plus_z3",
    }


def _exp2932_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete:citation_field_verifier_ready",
        "citation_verifier_ready": True,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        "field_match_accuracy": 0.85,
        "hallucination_detection_accuracy": 1.0,
        "inference_substrate": "live_llm_inference_plus_deterministic_verifier",
    }


def _exp2933_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: kan_rbf_importance_self_learning_passed",
        "kan_cl_self_learning_ready": True,
        "continuous_self_learning_targeted": True,
        "utility_delta_vs_replay_only": 0.5,
        "energy_proxy_delta": 0.4958,
        "forgetting_rate": 0.0,
        "forgetting_threshold": 0.05,
        "non_forgetting_passed": True,
        "updated_knot_or_rbf_count": 12,
        "inference_substrate": "local_training_simulation",
    }


def _exp2934_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: reformulated and exact-verified",
        "reformulation_pipeline_ready": True,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        "final_feasibility_rate": 1.0,
        "final_optimality_rate": 1.0,
        "prefix_bound_available": False,
        "inference_substrate": "live_llm_inference_plus_exact_verifier",
    }


def _exp2935_payload(*, matrix_ready: bool = True, kv260_clean: bool = True) -> dict[str, Any]:
    clean_rows = [
        "corpus:FoVer",
        "corpus:HaluEval_FEVER",
        "exp2910_sota_codegen",
        "exp2913_kv260_claim_boundary",
        "exp2918_fr11_process_rewards",
        "exp2920_state_verifier_harness",
        "exp2924_aggregation_metadata_corrigendum",
        "exp2925_taxonomy_corrigendum",
        "exp2926_constraintbench_corrigendum",
        "exp2933_kan_cl_self_learning",
    ]
    if not kv260_clean:
        clean_rows.remove("exp2913_kv260_claim_boundary")
    headline_rows = [
        "corpus:FoVer",
        "corpus:HaluEval_FEVER",
        "exp2910_sota_codegen",
        "exp2913_kv260_claim_boundary",
        "exp2918_fr11_process_rewards",
        "exp2920_state_verifier_harness",
        "exp2926_constraintbench_corrigendum",
        "exp2933_kan_cl_self_learning",
    ]
    counts = {
        "clean": 14,
        "flagged": 4,
        "blocked": 4,
        "missing": 2,
        "projection_only": 1,
        "diagnostic_only": 2,
        "pilot_only": 3,
    }
    return {
        "honest_verdict": "complete: matrix_v10_ready=true",
        "matrix_v10_ready": matrix_ready,
        "matrix_v10_paper_boundary_ready": matrix_ready,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "METHODOLOGY_MISSING", "severity": "warn"}],
        "adversarial_audit_rerun": {"flagged": False, "findings": []},
        "row_classification_counts": counts,
        "headline_eligible_rows": headline_rows,
        "clean_rows": clean_rows,
        "flagged_rows": [
            "exp2911_code_hallucination_verifier",
            "exp2919_constraintbench_mini",
            "exp2932_citation_field_verifier",
            "exp2934_reformulation_pipeline",
        ],
        "blocked_rows": [
            "exp2914_gatemate_toolchain",
            "exp2927_gatemate_himbaechel_preflight",
            "exp2929_gatemate_flash_timing_boundary",
            "exp2931_llmeval_logic_z3_mini",
        ],
        "missing_rows": ["exp2915_gatemate_bitstream", "exp2928_gatemate_bitstream"],
        "projection_only_rows": ["exp2930_kv260_scaling_projection"],
        "diagnostic_only_rows": ["exp2916_thrml_parity", "exp2917_spilled_energy_micro_panel"],
        "pilot_only_rows": ["corpus:MBPP", "corpus:HumanEval", "exp2891_cctu"],
        "paper_claim_boundary": {
            "ready": matrix_ready,
            "headline_eligible_rows": headline_rows,
            "headline_claims": {
                "exp2913_kv260_claim_boundary": "KV260 matched n=64 sparse Ising speedup.",
                "exp2933_kan_cl_self_learning": (
                    "KAN update utility_delta_vs_replay_only=0.5 with forgetting_rate=0.0."
                ),
            },
            "supporting_paper_claim_rows": ["exp2925_taxonomy_corrigendum"],
        },
        "no_new_llm_call": True,
        "no_new_hardware_run": True,
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _write_scenario_sources(
    root: Path,
    *,
    matrix_ready: bool = True,
    kv260_clean: bool = True,
) -> None:
    payloads = {
        exp2936.EXPECTED_ARTIFACTS["exp2923"].path: _exp2923_payload(),
        exp2936.EXPECTED_ARTIFACTS["exp2924"].path: _exp2924_payload(),
        exp2936.EXPECTED_ARTIFACTS["exp2925"].path: _exp2925_payload(),
        exp2936.EXPECTED_ARTIFACTS["exp2926"].path: _exp2926_payload(),
        exp2936.EXPECTED_ARTIFACTS["exp2927"].path: _exp2927_payload(),
        exp2936.EXPECTED_ARTIFACTS["exp2929"].path: _exp2929_payload(),
        exp2936.EXPECTED_ARTIFACTS["exp2930"].path: _exp2930_payload(),
        exp2936.EXPECTED_ARTIFACTS["exp2931"].path: _exp2931_payload(),
        exp2936.EXPECTED_ARTIFACTS["exp2932"].path: _exp2932_payload(),
        exp2936.EXPECTED_ARTIFACTS["exp2933"].path: _exp2933_payload(),
        exp2936.EXPECTED_ARTIFACTS["exp2934"].path: _exp2934_payload(),
        exp2936.EXPECTED_ARTIFACTS["exp2935"].path: _exp2935_payload(
            matrix_ready=matrix_ready,
            kv260_clean=kv260_clean,
        ),
    }
    for rel_path, payload in payloads.items():
        _write_json(root, rel_path, payload)


def test_req_report_2936_spec_is_declared() -> None:
    """REQ-REPORT-2936: OpenSpec declares the .276 capstone contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-2936" in spec
    assert "SCENARIO-REPORT-2936" in spec
    assert str(exp2936.OUTPUT_REL_PATH) in spec


def test_scenario_report_2936_synthesizes_terminal_capstone(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2936: .276 closes honestly with missing GateMate preserved."""

    _write_scenario_sources(tmp_path)

    artifact = exp2936.build_artifact(tmp_path, started_s=10.0, now_s=12.5)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["milestone"] == "2026.05.276"
    assert artifact["paper_ready"] is True
    assert artifact["hardware_speedup_claim_eligible"] is True
    assert artifact["gate_mate_speedup_claim_eligible"] is False
    assert artifact["evidence_boundary_repaired"] is True
    assert artifact["sota_structured_generation_clean"] is False
    assert artifact["fr11_self_learning_clean"] is True
    assert artifact["no_new_llm_call"] is True
    assert artifact["no_new_hardware_run"] is True
    assert artifact["inference_substrate"] == exp2936.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["run_date"] == "20260523"

    assert artifact["clean_artifacts"] == [
        "exp2923",
        "exp2924",
        "exp2925",
        "exp2926",
        "exp2933",
        "exp2935",
    ]
    assert artifact["flagged_artifacts"] == ["exp2932", "exp2934"]
    assert artifact["blocked_artifacts"] == ["exp2927", "exp2929", "exp2931"]
    assert artifact["missing_artifacts"] == ["exp2928"]
    assert artifact["projection_only_artifacts"] == ["exp2930"]
    assert artifact["diagnostic_only_artifacts"] == []
    assert artifact["pilot_only_artifacts"] == []

    assert artifact["row_classification_counts"] == _exp2935_payload()["row_classification_counts"]
    assert artifact["row_boundaries"]["flagged_rows"] == _exp2935_payload()["flagged_rows"]
    assert artifact["row_boundaries"]["missing_rows"] == [
        "exp2915_gatemate_bitstream",
        "exp2928_gatemate_bitstream",
    ]

    evidence = artifact["evidence_boundary_summary"]
    assert evidence["aggregation_metadata_clean"] is True
    assert evidence["taxonomy_corrigendum_clean"] is True
    assert evidence["constraintbench_non_tautological"] is True
    assert evidence["constraintbench_duration_s"] == pytest.approx(68.1)

    gatemate = artifact["gate_mate_status"]
    assert gatemate["corrected_preflight_ready"] is True
    assert gatemate["constraints_ready"] is False
    assert gatemate["bitstream_artifact_present"] is False
    assert gatemate["flash_smoke_ready"] is False
    assert gatemate["exact_blocker"] == "missing exp2928 artifact"

    fr11 = artifact["continuous_self_learning_status"]
    assert fr11["kan_cl_self_learning_ready"] is True
    assert fr11["utility_delta_vs_replay_only"] == pytest.approx(0.5)
    assert fr11["forgetting_rate"] == pytest.approx(0.0)
    assert fr11["non_forgetting_passed"] is True

    assert (
        artifact["source_artifact_checksums"][str(exp2936.EXPECTED_ARTIFACTS["exp2928"].path)]
        is None
    )
    assert artifact["source_artifact_checksums"][
        str(exp2936.EXPECTED_ARTIFACTS["exp2935"].path)
    ] == _checksum(tmp_path / exp2936.EXPECTED_ARTIFACTS["exp2935"].path)
    assert len(artifact["top_three_next_actions"]) == 3
    assert "MMD" in artifact["top_three_next_actions"][0]
    assert "same-schedule" in artifact["top_three_next_actions"][1]
    assert "AUPRC" in artifact["top_three_next_actions"][2]


def test_req_report_2936_claims_close_when_matrix_headline_is_not_clean(tmp_path: Path) -> None:
    """REQ-REPORT-2936: KV260 speedup and paper readiness need a clean headline row."""

    _write_scenario_sources(tmp_path, kv260_clean=False)

    artifact = exp2936.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["paper_ready"] is False
    assert artifact["hardware_speedup_claim_eligible"] is False
    assert artifact["gate_mate_speedup_claim_eligible"] is False
    assert artifact["hardware_claim_boundary"]["kv260_prior_evidence_eligible"] is False


def test_req_report_2936_matrix_gate_failure_blocks_paper_ready(tmp_path: Path) -> None:
    """REQ-REPORT-2936: matrix-v10 gate failure prevents paper readiness."""

    _write_scenario_sources(tmp_path, matrix_ready=False)

    artifact = exp2936.build_artifact(tmp_path, started_s=2.0, now_s=2.4)

    assert artifact["paper_ready"] is False
    assert artifact["hardware_speedup_claim_eligible"] is True
    assert artifact["paper_claim_boundary"]["ready"] is False


def test_req_report_2936_write_artifact_persists_json(tmp_path: Path) -> None:
    """REQ-REPORT-2936: write_artifact persists the capstone JSON."""

    _write_scenario_sources(tmp_path)

    out = exp2936.write_artifact(tmp_path, started_s=0.0, now_s=0.125)

    assert out == tmp_path / exp2936.OUTPUT_REL_PATH
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["schema"] == exp2936.SCHEMA
    assert artifact["artifact"] == exp2936.ARTIFACT
    assert artifact["duration_s"] == pytest.approx(0.125)


def test_req_report_2936_defensive_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-2936: defensive parsing and classification fail closed."""

    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert exp2936.read_json_mapping(bad) == {}
    assert exp2936.read_json_mapping(tmp_path / "missing.json") == {}
    array = tmp_path / "array.json"
    array.write_text("[1]", encoding="utf-8")
    assert exp2936.read_json_mapping(array) == {}

    assert exp2936.classify_artifact("exp2928", {}, False) == "missing"
    assert exp2936.classify_artifact("exp2930", _exp2930_payload(), True) == "projection_only"
    assert (
        exp2936.classify_artifact(
            "exp2923",
            {"honest_verdict": "complete: diagnostic", "diagnostic_only": True},
            True,
        )
        == "diagnostic_only"
    )
    assert (
        exp2936.classify_artifact(
            "exp2923",
            {"honest_verdict": "complete: pilot", "pilot_only": True},
            True,
        )
        == "pilot_only"
    )
    assert exp2936.classify_artifact("exp2925", _exp2925_payload(), True) == "clean"
    assert (
        exp2936.classify_artifact(
            "exp2925",
            {**_exp2925_payload(), "taxonomy_corrigendum_clean": False},
            True,
        )
        == "flagged"
    )
    assert exp2936.classify_artifact("exp2932", _exp2932_payload(), True) == "flagged"
    assert exp2936.classify_artifact("exp2931", _exp2931_payload(), True) == "blocked"
    assert (
        exp2936.classify_artifact(
            "exp2923",
            {"honest_verdict": "partial: archive", "archive_ready": True},
            True,
        )
        == "blocked"
    )
    assert exp2936._has_current_flags({"adversarial_verify_passed": False}) is True
    assert exp2936._has_current_flags({"adversarial_audit_rerun": {"flagged": True}}) is True
    assert exp2936._has_current_flags({"adversarial_verify_flags": [{"kind": "X"}]}) is True
    assert exp2936._has_current_flags({"adversarial_verify_summary": {"flag_count": 1}}) is True
    assert (
        exp2936._paper_ready(
            {
                **_exp2935_payload(),
                "matrix_v10_paper_boundary_ready": False,
            }
        )
        is False
    )
    assert (
        exp2936._paper_ready(
            {
                **_exp2935_payload(),
                "paper_claim_boundary": {"ready": False},
            }
        )
        is False
    )
    assert exp2936._matrix_row_counts({}) == {row_class: 0 for row_class in exp2936.ROW_CLASSES}
