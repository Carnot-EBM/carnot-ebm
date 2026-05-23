"""Tests for Exp 2921 cross-corpus matrix v9 and paper boundary.

Spec refs: REQ-REPORT-2921, SCENARIO-REPORT-2921.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v9_2921 as exp2921


def _write_json(root: Path, rel_path: str | Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(root: Path, rel_path: str | Path) -> str:
    return hashlib.sha256((root / rel_path).read_bytes()).hexdigest()


def _v8_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_2902_cross_corpus_matrix_v8",
        "honest_verdict": (
            "complete: cross-corpus matrix v8 aggregated with forward-only provenance; "
            "clean=6; flagged=2; blocked=0; pilot_only=3"
        ),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "rows_clean": [
            "corpus:FoVer",
            "corpus:HaluEval_FEVER",
            "corpus:TruthfulQA",
            "exp2890_code_structural_dependency",
            "exp2892_vericot",
            "exp2898_kv260_hardware",
        ],
        "rows_flagged": ["corpus:MBPP", "corpus:HumanEval"],
        "rows_blocked": [],
        "rows_pilot_only": ["corpus:MBPP", "corpus:HumanEval", "exp2891_cctu"],
        "matrix_rows": [
            {
                "row_id": "corpus:FoVer",
                "row_label": "FoVer",
                "row_kind": "v7_corpus_row",
                "row_status": "clean",
                "flag_reasons": [],
                "summary": {"headline_eligible": True, "primary_metric": {"auroc": 0.913}},
            },
            {
                "row_id": "corpus:HaluEval_FEVER",
                "row_label": "HaluEval/FEVER",
                "row_kind": "v7_corpus_row",
                "row_status": "clean",
                "flag_reasons": [],
                "summary": {"headline_eligible": True, "primary_metric": {"auroc": 0.553}},
            },
            {
                "row_id": "corpus:MBPP",
                "row_label": "MBPP",
                "row_kind": "v7_corpus_row",
                "row_status": "pilot_only_flagged_support",
                "flag_reasons": ["flagged_adversarial=true"],
                "summary": {"pilot_only": True},
            },
            {
                "row_id": "corpus:HumanEval",
                "row_label": "HumanEval",
                "row_kind": "v7_corpus_row",
                "row_status": "pilot_only_flagged_support",
                "flag_reasons": ["flagged_adversarial=true"],
                "summary": {"pilot_only": True},
            },
            {
                "row_id": "corpus:TruthfulQA",
                "row_label": "TruthfulQA",
                "row_kind": "v7_corpus_row",
                "row_status": "clean",
                "flag_reasons": [],
                "summary": {"taxonomy_only": True},
            },
            {
                "row_id": "exp2890_code_structural_dependency",
                "row_label": "Code Structural Dependency",
                "row_kind": "support_artifact_row",
                "row_status": "clean",
                "summary": {"n_rows_verified": 20},
            },
            {
                "row_id": "exp2891_cctu",
                "row_label": "CCTU executable constraint pilot",
                "row_kind": "support_artifact_row",
                "row_status": "pilot_only",
                "summary": {"n_cases": 5},
            },
            {
                "row_id": "exp2892_vericot",
                "row_label": "VeriCoT exact frontier",
                "row_kind": "support_artifact_row",
                "row_status": "clean",
                "summary": {"n_vericot_supported_rows": 25},
            },
            {
                "row_id": "exp2898_kv260_hardware",
                "row_label": "KV260 Ising hardware latency",
                "row_kind": "support_artifact_row",
                "row_status": "clean",
                "summary": {"inference_substrate": "hardware_smoke"},
            },
        ],
    }


def _exp2910_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_2910_sota_code_generation_corrigendum_v2",
        "honest_verdict": "complete: SOTA code-generation corrigendum executed",
        "inference_substrate": "live_llm_inference",
        "codegen_corrigendum_ready": True,
        "candidate_generation_clean": True,
        "aggregate_pass_at_1": 0.075,
        "aggregate_pass_at_k": 0.175,
        "pass_at_k_exceeds_pass_at_1": True,
        "n_tasks_per_corpus": 20,
        "k_candidates_per_task": 8,
        "model_specs": [{"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF"}],
        "random_seed": 2910,
        "reproducibility_checksum": "abc123",
        "flagged_adversarial": False,
        "corrigendum_pending": None,
    }


def _exp2911_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_2911_code_hallucination_taxonomy_verifier_v1",
        "honest_verdict": "complete: Exp 2910 code candidates labeled",
        "inference_substrate": "deterministic_verifier",
        "code_hallucination_verifier_ready": True,
        "pass_rate_after_taxonomy_filter": 0.0967,
        "syntax_error_rate": 0.38125,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
    }


def _exp2912_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: same_basis_cpu_gibbs_baseline_ready_no_speedup_claim",
        "inference_substrate": "cpu_sampler",
        "same_basis_cpu_baseline_ready": True,
        "speedup_claim_made": False,
    }


def _exp2913_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: kv260_same_basis_hardware_cpu_speedup_claim_eligible",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "kv260_claim_boundary_ready": True,
        "same_basis_verified": True,
        "hardware_speedup_claim_eligible": True,
        "speedup_claim_made": True,
        "matrix_row_candidate": {
            "eligible_for_matrix_v9": True,
            "eligible_for_paper_v6": True,
            "speedup_ratio_median_by_sample_count": {"100": 19.13},
        },
        "paper_claim_boundary": "Bounded n=64 KV260/CPU speedup claim only.",
    }


def _exp2914_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "blocked_gatemate_toolchain_missing",
        "inference_substrate": "hardware_preflight",
        "gatemate_toolchain_ready": False,
        "missing_toolchain": ["nextpnr-gatemate"],
        "no_flash_attempted": True,
    }


def _exp2916_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: thrml_kv260_n64_simulator_parity_ready_no_hardware_claim",
        "inference_substrate": "simulator_parity",
        "thrml_kv260_parity_ready": True,
        "no_tsu_hardware_claim": True,
        "hardware_claims_made": {"tsu": False, "new_latency": False},
    }


def _exp2917_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_2917_spilled_energy_logit_detector_micro_panel_v1",
        "honest_verdict": "complete: spilled_energy_micro_panel_diagnostic_ready",
        "inference_substrate": "live_llm_inference",
        "spilled_energy_micro_panel_ready": True,
        "benchmark_claim_made": False,
        "claim_boundary": "diagnostic_only_no_benchmark_claim",
        "separation_summary": {"available": True, "n_examples": 24},
    }


def _exp2918_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_2918_fr11_verifiable_process_rewards_self_learning_v1",
        "honest_verdict": "complete: verifier_process_rewards_updated_replay_scheduler",
        "inference_substrate": "deterministic_verifier_plus_replay",
        "online_self_learning_ready": True,
        "online_update_performed": True,
        "replay_scheduler_updated": True,
        "model_weights_mutated": False,
        "forgetting_rate": 0.0,
        "delta_overall": 0.8559,
        "hardware_replay_used": True,
    }


def _exp2919_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_2919_constraintbench_mini_direct_optimization_v1",
        "honest_verdict": "complete: constraintbench mini direct optimization measured",
        "inference_substrate": "live_llm_inference_plus_exact_verifier",
        "constraintbench_mini_ready": True,
        "syntax_valid_rate": 0.277778,
        "feasibility_rate": 0.277778,
        "optimality_rate": 0.111111,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
    }


def _exp2920_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_2920_opencomputer_style_state_verifier_harness_v1",
        "honest_verdict": "complete: deterministic OpenComputer-style state verifier harness ready",
        "inference_substrate": "deterministic_state_verifier",
        "state_verifier_harness_ready": True,
        "n_state_tasks": 4,
        "llm_judge_used": False,
        "golden_state_pass_rate": 1.0,
        "negative_state_reject_rate": 1.0,
    }


def _write_ready_sources(root: Path) -> None:
    payloads = {
        exp2921.MATRIX_V8_REL_PATH: _v8_payload(),
        exp2921.EXP2910_REL_PATH: _exp2910_payload(),
        exp2921.EXP2911_REL_PATH: _exp2911_payload(),
        exp2921.EXP2912_REL_PATH: _exp2912_payload(),
        exp2921.EXP2913_REL_PATH: _exp2913_payload(),
        exp2921.EXP2914_REL_PATH: _exp2914_payload(),
        exp2921.EXP2916_REL_PATH: _exp2916_payload(),
        exp2921.EXP2917_REL_PATH: _exp2917_payload(),
        exp2921.EXP2918_REL_PATH: _exp2918_payload(),
        exp2921.EXP2919_REL_PATH: _exp2919_payload(),
        exp2921.EXP2920_REL_PATH: _exp2920_payload(),
    }
    for rel_path, payload in payloads.items():
        _write_json(root, rel_path, payload)


def _rows_by_id(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["row_id"]: row for row in artifact["matrix_rows"]}


def test_req_report_2921_spec_is_declared() -> None:
    """REQ-REPORT-2921: OpenSpec declares the v9 artifact contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-2921" in spec
    assert "SCENARIO-REPORT-2921" in spec
    assert "blocked_gate_inconsistent" in spec


def test_scenario_report_2921_builds_v9_and_paper_boundary(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2921: only clean bounded rows become headline eligible."""

    _write_ready_sources(tmp_path)

    artifact = exp2921.build_artifact(tmp_path, started_s=10.0, now_s=12.5)

    required = {
        "honest_verdict",
        "cross_corpus_matrix_v9_built",
        "paper_claim_boundary_ready",
        "headline_eligible_rows",
        "clean_rows",
        "flagged_rows",
        "blocked_rows",
        "pilot_only_rows",
        "diagnostic_only_rows",
        "missing_rows",
        "matrix_v9_path",
        "paper_v6_claim_boundary",
        "inference_substrate",
        "duration_s",
        "run_date",
    }
    assert required <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["cross_corpus_matrix_v9_built"] is True
    assert artifact["paper_claim_boundary_ready"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["run_date"] == "20260523"
    assert artifact["matrix_v9_path"] == str(exp2921.OUTPUT_REL_PATH)

    assert artifact["headline_eligible_rows"] == [
        "corpus:FoVer",
        "corpus:HaluEval_FEVER",
        "exp2910_sota_codegen",
        "exp2913_kv260_claim_boundary",
        "exp2918_fr11_process_rewards",
        "exp2920_state_verifier_harness",
    ]
    assert "exp2911_code_hallucination_verifier" in artifact["flagged_rows"]
    assert "exp2919_constraintbench_mini" in artifact["flagged_rows"]
    assert "exp2914_gatemate_toolchain" in artifact["blocked_rows"]
    assert artifact["diagnostic_only_rows"] == [
        "exp2916_thrml_parity",
        "exp2917_spilled_energy_micro_panel",
    ]
    assert artifact["missing_rows"] == ["exp2915_gatemate_bitstream"]
    assert "exp2920_state_verifier_harness" in artifact["clean_rows"]

    rows = _rows_by_id(artifact)
    assert rows["exp2910_sota_codegen"]["row_status"] == "clean"
    assert rows["exp2910_sota_codegen"]["headline_eligible"] is True
    assert rows["exp2911_code_hallucination_verifier"]["row_status"] == "flagged"
    assert rows["exp2911_code_hallucination_verifier"]["headline_eligible"] is False
    assert rows["exp2913_kv260_claim_boundary"]["headline_eligible"] is True
    assert rows["exp2916_thrml_parity"]["row_status"] == "diagnostic_only"
    assert rows["exp2917_spilled_energy_micro_panel"]["row_status"] == "diagnostic_only"
    assert rows["exp2919_constraintbench_mini"]["row_status"] == "flagged"

    boundary = artifact["paper_v6_claim_boundary"]
    assert boundary["ready"] is True
    assert boundary["headline_eligible_rows"] == artifact["headline_eligible_rows"]
    assert boundary["headline_claims"]["exp2913_kv260_claim_boundary"].startswith("Bounded")
    assert boundary["non_headline_rows"]["exp2911_code_hallucination_verifier"]["status"] == (
        "flagged"
    )
    assert boundary["non_headline_rows"]["exp2916_thrml_parity"]["status"] == "diagnostic_only"
    assert boundary["non_headline_rows"]["exp2914_gatemate_toolchain"]["status"] == "blocked"
    assert boundary["non_headline_rows"]["exp2915_gatemate_bitstream"]["status"] == "missing"

    citations = {item["experiment_id"]: item for item in artifact["cited_upstream_artifacts"]}
    assert citations["exp2902"]["sha256"] == _sha256(tmp_path, exp2921.MATRIX_V8_REL_PATH)
    assert citations["exp2910"]["sha256"] == _sha256(tmp_path, exp2921.EXP2910_REL_PATH)
    assert citations["exp2915"]["sha256"] is None
    for row in artifact["matrix_rows"]:
        if row["source_artifact"] != str(exp2921.EXP2915_REL_PATH):
            assert row["source_sha256"] == _sha256(tmp_path, row["source_artifact"])

    out = exp2921.write_artifact(tmp_path, started_s=1.0, now_s=1.25)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert out == tmp_path / exp2921.OUTPUT_REL_PATH
    assert saved["duration_s"] == pytest.approx(0.25)


def test_req_report_2921_blocks_when_gate_failed_open(tmp_path: Path) -> None:
    """REQ-REPORT-2921: an absent or not-ready gated artifact blocks v9."""

    _write_ready_sources(tmp_path)
    _write_json(
        tmp_path,
        exp2921.EXP2919_REL_PATH,
        {**_exp2919_payload(), "constraintbench_mini_ready": False},
    )

    artifact = exp2921.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["honest_verdict"] == "blocked_gate_inconsistent"
    assert artifact["cross_corpus_matrix_v9_built"] is False
    assert artifact["paper_claim_boundary_ready"] is False
    assert artifact["headline_eligible_rows"] == []
    assert artifact["blocked_rows"] == ["exp2919_constraintbench_mini"]
    assert artifact["gate_errors"] == [
        {
            "experiment_id": "exp2919",
            "row_id": "exp2919_constraintbench_mini",
            "artifact_path": str(exp2921.EXP2919_REL_PATH),
            "required_field": "constraintbench_mini_ready",
            "actual_value": False,
        }
    ]
    assert artifact["paper_v6_claim_boundary"]["ready"] is False

    missing_root = tmp_path / "missing_gate"
    _write_json(missing_root, exp2921.MATRIX_V8_REL_PATH, _v8_payload())
    missing = exp2921.build_artifact(missing_root)
    assert missing["honest_verdict"] == "blocked_gate_inconsistent"
    assert "exp2911_code_hallucination_verifier" in missing["missing_rows"]


def test_req_report_2921_blocks_without_matrix_v8_and_handles_bad_json(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2921: matrix v8 is required after gates are consistent."""

    assert exp2921.read_json(tmp_path / "absent.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert exp2921.read_json(bad) == {}
    array = tmp_path / "array.json"
    array.write_text("[1]", encoding="utf-8")
    assert exp2921.read_json(array) == {}

    for rel_path, payload in {
        exp2921.EXP2911_REL_PATH: _exp2911_payload(),
        exp2921.EXP2913_REL_PATH: _exp2913_payload(),
        exp2921.EXP2918_REL_PATH: _exp2918_payload(),
        exp2921.EXP2919_REL_PATH: _exp2919_payload(),
        exp2921.EXP2920_REL_PATH: _exp2920_payload(),
    }.items():
        _write_json(tmp_path, rel_path, payload)

    artifact = exp2921.build_artifact(tmp_path, started_s=3.0, now_s=3.25)

    assert artifact["honest_verdict"] == "blocked_matrix_v8_missing"
    assert artifact["cross_corpus_matrix_v9_built"] is False
    assert artifact["paper_claim_boundary_ready"] is False
    assert artifact["blocked_rows"] == ["exp2902_matrix_v8"]
    assert artifact["matrix_rows"] == []


def test_req_report_2921_defensive_classification_helpers() -> None:
    """REQ-REPORT-2921: defensive branches preserve non-headline boundaries."""

    assert exp2921._classify_v8_status("blocked", []) == "blocked"
    assert exp2921._classify_v8_status("diagnostic_only", []) == "diagnostic_only"
    assert exp2921._candidate_status("exp2911", {"code_hallucination_verifier_ready": True}) == (
        "clean"
    )
    assert exp2921._candidate_status("exp2914", {"gatemate_toolchain_ready": True}) == "clean"
    assert exp2921._candidate_status("exp2915", {"gatemate_bitstream_built": True}) == "clean"
    assert exp2921._candidate_status("exp2919", {"constraintbench_mini_ready": True}) == "clean"
    assert exp2921._candidate_status("exp9999", {"honest_verdict": "complete: unknown"}) == (
        "blocked"
    )
    assert exp2921._candidate_claim_boundary("unknown_clean_row", {}, True) == ""
    assert exp2921._has_flags({"adversarial_verify_passed": False}) is True
    assert exp2921._has_flags({"corrigendum_pending": [{"kind": "METHODOLOGY_MISSING"}]}) is True
    assert exp2921._flag_reasons(
        {
            "adversarial_verify_passed": False,
            "corrigendum_pending": [{"kind": "METHODOLOGY_MISSING"}],
            "adversarial_verify_flags": [{"kind": "TAUTOLOGY"}],
            "adversarial_verify_summary": {"flag_count": 1},
        }
    ) == [
        "adversarial_verify_passed=false",
        "METHODOLOGY_MISSING:unknown",
        "adversarial_verify_flags_present",
        "adversarial_verify_summary_flag_count",
    ]
