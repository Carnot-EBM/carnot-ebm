"""Tests for Exp 3308 quality-flag root-cause autopsy.

Spec refs: REQ-REPORT-3308, SCENARIO-REPORT-3308.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import quality_flag_root_cause_autopsy_3308 as mod


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _flag(kind: str, severity: str, detail: str) -> dict[str, str]:
    return {"kind": kind, "severity": severity, "detail": detail}


def _exp3300_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3300_full_garak_dataflip_gate_rerun_v3",
        "experiment_id": "exp3300",
        "garak_redteam_eval_v3_ready": True,
        "garak_gate_passed": True,
        "dataflip_gate_passed": False,
        "attack_success_rate": 0.0,
        "aligned_instruction_false_positive_rate": 0.033333,
        "refusal_rate": 0.033333,
        "error_count": 0,
        "duration_s": 39.796248,
        "inference_substrate": "llama_cpp_gpu_openai_adapter",
        "gpu_mem_used_mib": 9072,
        "tokens_generated": 2400,
        "model_specs": {
            "runtime": "llama_cpp_openai_compatible_rest",
            "local_target_adapter": "defended_openai_compatible_llama_cpp",
            "mandated_model_ids": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
                "unsloth/gemma-4-26B-A4B-it-GGUF",
            ],
        },
        "models_used": [
            {
                "model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "model_path": "/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
                "live_target_call": True,
                "local_target_adapter_started": True,
                "gpu_mem_used_mib": 9072,
                "tokens_generated": 2400,
            }
        ],
        "per_probe_family_metrics": {
            "aligned_benign": {
                "probe_count": 30,
                "refusal_count": 1,
                "refusal_rate": 0.033333,
                "false_positive_count": 1,
                "false_positive_rate": 0.033333,
            },
            "dataflip_kad_adaptive": {
                "probe_count": 30,
                "refusal_count": 2,
                "detection_count": 2,
                "detection_rate": 0.066667,
                "attack_success_count": 28,
                "attack_success_rate": 0.933333,
            },
            "jailbreak_encoding": {"probe_count": 30, "refusal_count": 1},
            "promptinject": {"probe_count": 60, "refusal_count": 1},
        },
        "preconditions_checked": [
            {"name": "nvidia_smi", "passed": True, "gpu_count": 2},
            {"name": "selected_python_cuda", "passed": True, "cuda_available": True},
            {"name": "local_gguf_cache", "passed": True, "cached_model_count": 1},
        ],
        "adapter_start_evidence": {
            "panel_duration_s": 38.288767,
            "raw_report_path": "results/experiment_3300_full_garak_dataflip_gate_rerun_v3_adapter_stderr.log",
        },
        "corrigendum_pending": [
            _flag(
                "TAUTOLOGY",
                "critical",
                "aligned_instruction_false_positive_rate=0.033333 and refusal_rate=0.033333 agree to >5 sig figs.",
            ),
            _flag("IMPLAUSIBLE_PERFECT", "info", "error_count=0.0 (exactly zero)."),
            _flag(
                "DURATION_TOO_SHORT",
                "critical",
                "duration_s=39.796248 but artifact references compute-bound markers (GGUF / CUDA / live model).",
            ),
        ],
        "honest_verdict": "complete: fixture",
    }


def _exp3302_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3302_headline_sota_repair_panel_v11",
        "experiment_id": "exp3302",
        "headline_repair_panel_ready": True,
        "repair_panel_ran": True,
        "headline_claim_allowed": False,
        "provenance_clean": False,
        "flagged_adversarial": True,
        "panel_case_count": 30,
        "verified_success_count": 27,
        "false_accept_count": 0,
        "abstention_count": 0,
        "repair_success_rate": 0.9,
        "duration_s": 15.496424,
        "inference_substrate": "live_local_sota_gguf_repair_plus_calibrated_clean_verifier",
        "gpu_mem_used_mib": 18152,
        "tokens_generated": 86,
        "model_specs": {
            "runtime": "llama_cpp_local_gguf_only",
            "generation_runtime": "llama_cpp_local_generation",
            "verification_runtime": "exp3287_calibrated_accept_reject_abstain_contract",
            "mandated_model_ids": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
                "unsloth/gemma-4-26B-A4B-it-GGUF",
            ],
        },
        "models_used": [
            {
                "model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "model_path": "/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
                "legacy_small_model": False,
                "gpu_mem_used_mib": 18152,
            }
        ],
        "missing_model_specs": [
            {"model_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "reason": "not_cached"},
            {"model_id": "unsloth/gemma-4-31B-it-GGUF", "reason": "not_cached"},
        ],
        "preconditions_checked": [
            {"name": "nvidia_smi", "passed": True, "gpu_count": 2},
            {"name": "selected_python_cuda", "passed": True, "cuda_available": True},
            {"name": "mandated_sota_gguf_cache", "passed": True},
        ],
        "corrigendum_pending": [
            _flag(
                "DURATION_TOO_SHORT",
                "critical",
                "duration_s=15.496424 but artifact references compute-bound markers (GGUF / CUDA / live model).",
            )
        ],
        "honest_verdict": "complete: fixture",
    }


def _exp3303_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3303_repair_headline_evidence_audit_v1",
        "experiment_id": "exp3303",
        "repair_headline_evidence_audit_ready": True,
        "headline_claim_allowed_after_audit": False,
        "source_headline_claim_allowed": False,
        "source_provenance_clean": False,
        "substrate_consistency_passed": False,
        "no_new_model_execution": True,
        "no_new_repair_generation": True,
        "duration_s": 0.001598,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "model_invocation_summary": {
            "used_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "missing_model_ids": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
            ],
            "mandated_model_ids": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-26B-A4B-it-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
            ],
            "actual_model_declarations_present": True,
            "legacy_small_model_used": False,
        },
        "adversarial_verify_flags": [
            _flag(
                "DURATION_TOO_SHORT",
                "critical",
                "duration_s=15.496424 but artifact references compute-bound markers (GGUF / CUDA / live model).",
            )
        ],
        "honest_verdict": "complete: fixture",
    }


def _stage_sources(root: Path) -> None:
    exp3300 = _exp3300_payload()
    exp3302 = _exp3302_payload()
    exp3303 = _exp3303_payload()
    _write_json(root, mod.EXP3300_REL_PATH, exp3300)
    _write_json(root, mod.EXP3302_REL_PATH, exp3302)
    _write_json(root, mod.EXP3303_REL_PATH, exp3303)
    _write_json(
        root,
        mod.EXP3305_REL_PATH,
        {
            "artifact": "experiment_3305_evidence_matrix_v37",
            "experiment_id": "exp3305",
            "matrix_v37_ready": True,
            "garak_gate_passed": True,
            "dataflip_gate_passed": False,
            "repair_headline_claim_allowed": False,
            "top_gap": "clear_garak_dataflip_and_quality_flags",
            "paper_blocker_count": 8,
            "evidence_rows": [
                {
                    "experiment_id": "exp3300",
                    "quality_flags": exp3300["corrigendum_pending"],
                    "critical_quality_flags": [exp3300["corrigendum_pending"][0], exp3300["corrigendum_pending"][2]],
                    "blocker_reasons": ["dataflip_gate_failed", "dataflip_gate_passed=false"],
                    "inference_substrate": exp3300["inference_substrate"],
                    "summary": {
                        "dataflip_gate_passed": False,
                        "garak_gate_passed": True,
                        "attack_success_rate": 0.0,
                    },
                },
                {
                    "experiment_id": "exp3302",
                    "quality_flags": exp3302["corrigendum_pending"],
                    "critical_quality_flags": exp3302["corrigendum_pending"],
                    "blocker_reasons": ["headline_claim_allowed=false", "provenance_clean=false"],
                    "inference_substrate": exp3302["inference_substrate"],
                    "summary": {
                        "headline_claim_allowed": False,
                        "provenance_clean": False,
                        "verified_success_count": 27,
                    },
                },
                {
                    "experiment_id": "exp3303",
                    "quality_flags": exp3303["adversarial_verify_flags"],
                    "critical_quality_flags": exp3303["adversarial_verify_flags"],
                    "blocker_reasons": [
                        "headline_claim_allowed_after_audit=false",
                        "source_headline_claim_allowed=false",
                        "source_provenance_clean=false",
                        "substrate_consistency_passed=false",
                    ],
                    "inference_substrate": exp3303["inference_substrate"],
                    "summary": {
                        "headline_claim_allowed_after_audit": False,
                        "source_provenance_clean": False,
                        "substrate_consistency_passed": False,
                    },
                },
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3306_REL_PATH,
        {
            "artifact": "experiment_3306_capstone_v305",
            "experiment_id": "exp3306",
            "capstone_v305_ready": True,
            "paper_ready": False,
            "publication_blocker_count": 8,
            "next_top_gap": "clear_garak_dataflip_and_quality_flags",
            "garak_gate_passed": True,
            "repair_headline_claim_allowed": False,
        },
    )


def test_req_report_3308_spec_anchor_declares_autopsy_schema() -> None:
    """REQ-REPORT-3308: OpenSpec names the autopsy contract before code."""

    spec = (mod.REPO_ROOT / mod.SPEC_REL_PATH).read_text(encoding="utf-8")

    assert "REQ-REPORT-3308" in spec
    assert "SCENARIO-REPORT-3308" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.EXP3300_REL_PATH.as_posix() in spec
    assert mod.EXP3302_REL_PATH.as_posix() in spec
    assert mod.EXP3303_REL_PATH.as_posix() in spec
    assert "scripts/research_conductor.py" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_3308_writes_quality_flag_autopsy(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3308: current flags become concrete rerun gates."""

    _stage_sources(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=10.0,
        now_s=13.25,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "results/out.json"
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["quality_flag_autopsy_ready"] is True
    assert artifact["no_new_model_execution"] is True
    assert artifact["no_new_garak_run"] is True
    assert artifact["no_new_repair_generation"] is True
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["honest_verdict"].startswith("complete:")

    garak_flags = [(flag["experiment_id"], flag["kind"], flag["severity"]) for flag in artifact["garak_quality_flags"]]
    assert garak_flags == [
        ("exp3300", "TAUTOLOGY", "critical"),
        ("exp3300", "IMPLAUSIBLE_PERFECT", "info"),
        ("exp3300", "DURATION_TOO_SHORT", "critical"),
    ]
    repair_flags = [(flag["experiment_id"], flag["kind"], flag["severity"]) for flag in artifact["repair_quality_flags"]]
    assert repair_flags == [
        ("exp3302", "DURATION_TOO_SHORT", "critical"),
        ("exp3303", "DURATION_TOO_SHORT", "critical"),
    ]

    hypotheses = {row["id"]: row for row in artifact["root_cause_hypotheses"]}
    tautology = hypotheses["exp3300_tautology_metric_lineage"]
    assert tautology["verdict"] == "derived_metric_coincidence_not_direct_calculation_conflation"
    assert tautology["observed_counts"]["aligned_benign_false_positive_count"] == 1
    assert tautology["observed_counts"]["aligned_benign_probe_count"] == 30
    assert tautology["observed_counts"]["total_refusal_count"] == 5
    assert tautology["observed_counts"]["total_probe_count"] == 150
    assert "top_level_refusal_rate" in tautology["metrics_involved"]
    assert "aligned_benign_false_positive_rate" in tautology["metrics_involved"]

    duration = hypotheses["duration_too_short_runtime_provenance"]
    exp3300_duration = next(row for row in duration["affected_artifacts"] if row["experiment_id"] == "exp3300")
    assert "inference_substrate=llama_cpp_gpu_openai_adapter" in exp3300_duration["live_model_markers"]
    assert "models_used.model_path=.gguf" in exp3300_duration["live_model_markers"]
    assert "model_load_start_end_timestamps" in exp3300_duration["missing_runtime_provenance"]
    assert "llama_cpp_load_stderr_excerpt_or_load_receipt" in exp3300_duration["missing_runtime_provenance"]

    repair = hypotheses["repair_substrate_provenance_blocker"]
    assert repair["panel_fields"]["inference_substrate"] == "live_local_sota_gguf_repair_plus_calibrated_clean_verifier"
    assert repair["audit_fields"]["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert repair["agreements_that_block_promotion"]["source_provenance_clean"] is False
    assert repair["agreements_that_block_promotion"]["source_headline_claim_allowed"] is False
    assert {
        "field": "inference_substrate",
        "panel": "live_local_sota_gguf_repair_plus_calibrated_clean_verifier",
        "audit": "aggregation_from_upstream_artifacts",
        "classification": "expected_audit_aggregation_boundary_not_a_live_rerun",
    } in repair["field_disagreements"]

    reruns = {row["experiment_id"]: row for row in artifact["rerun_requirements"]}
    assert set(reruns) == {"exp3309", "exp3312", "exp3316"}
    assert "minimum_live_duration_s" in reruns["exp3309"]["acceptance_requirements"]
    assert "metric_numerator_denominator_lineage" in reruns["exp3309"]["acceptance_requirements"]
    assert "dataflip_gate_passed=true" in reruns["exp3312"]["acceptance_requirements"]
    assert "quality_flags_cleared=true" in reruns["exp3312"]["acceptance_requirements"]
    assert "substrate_consistency_passed=true" in reruns["exp3316"]["acceptance_requirements"]
    assert "headline_claim_allowed=true_or_honestly_blocked" in reruns["exp3316"]["acceptance_requirements"]

    analyzed = {row["experiment_id"]: row for row in artifact["analyzed_artifacts"]}
    assert analyzed["exp3305"]["ready"] is True
    assert analyzed["exp3306"]["ready"] is True
    mod.validate_artifact(artifact)


def test_req_report_3308_validate_rejects_incomplete_artifacts(tmp_path: Path) -> None:
    """REQ-REPORT-3308: malformed autopsies cannot masquerade as complete."""

    _stage_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="quality_flag_autopsy_ready"):
        mod.validate_artifact(artifact | {"quality_flag_autopsy_ready": "true"})
    with pytest.raises(ValueError, match="garak_quality_flags"):
        mod.validate_artifact(artifact | {"garak_quality_flags": []})
    with pytest.raises(ValueError, match="repair_quality_flags"):
        mod.validate_artifact(artifact | {"repair_quality_flags": []})
    with pytest.raises(ValueError, match="root_cause_hypotheses"):
        mod.validate_artifact(artifact | {"root_cause_hypotheses": []})
    with pytest.raises(ValueError, match="rerun_requirements"):
        mod.validate_artifact(artifact | {"rerun_requirements": []})
    with pytest.raises(ValueError, match="no_new_model_execution"):
        mod.validate_artifact(artifact | {"no_new_model_execution": False})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})


def test_req_report_3308_fallback_flag_sources_and_defensive_helpers() -> None:
    """REQ-REPORT-3308: source fallback logic keeps flags auditable."""

    corr_flags = mod.quality_flags_for(
        "exp3302",
        {"corrigendum_pending": [_flag("DURATION_TOO_SHORT", "critical", "from panel")]},
        {},
    )
    audit_flags = mod.quality_flags_for(
        "exp3303",
        {"adversarial_verify_flags": [_flag("DURATION_TOO_SHORT", "critical", "from audit")]},
        {},
    )
    duration = mod.duration_root_cause(
        {
            "exp3300": {},
            "exp3302": {"corrigendum_pending": corr_flags, "duration_s": 12.0},
            "exp3303": {},
        }
    )

    assert corr_flags[0]["source"] == "exp3302.corrigendum_pending"
    assert audit_flags[0]["source"] == "exp3303.adversarial_verify_flags"
    assert [row["experiment_id"] for row in duration["affected_artifacts"]] == ["exp3302"]
    assert mod.string_list(None) == []
    assert mod.string_list("not-a-list") == []
    assert mod.string_list(42) == []
