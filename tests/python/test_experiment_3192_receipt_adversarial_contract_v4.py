"""Tests for Exp 3192 receipt/adversarial contract v4.

Spec refs: REQ-VERIFY-3192, SCENARIO-VERIFY-3192.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import receipt_adversarial_contract_v4 as mod


REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "contract_version",
    "proof_execution_required_fields",
    "clean_rerun_required_fields",
    "aggregate_required_fields",
    "gated_skip_required_fields",
    "accepted_substrate_classes",
    "rejected_headline_substrate_classes",
    "terminal_verdict_prefixes",
    "blocked_verdict_prefixes",
    "downstream_unlock_fields",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "source\n") -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(index: int) -> dict[str, Any]:
    return {
        "selected_model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "model_path": f"/models/gemma4-{index}.gguf",
        "model_file_hash": f"model-hash-{index}",
        "loader_name": "llama_cpp.Llama",
        "substrate_used": "cpu_fallback_receipt_only",
        "prompt_hash": f"prompt-hash-{index}",
        "transcript_hash": f"transcript-hash-{index}",
        "token_counts": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10},
        "random_seed": 20260527 + index,
        "wall_clock_s": 7.5 + index,
        "command_hash": f"command-hash-{index}",
        "subprocess_return_code": 0,
        "stderr_tail": "llama.cpp fixture stderr tail",
        "throughput_plausibility": True,
        "replay_count": index + 1,
        "worker_code_sha256": f"worker-hash-{index}",
    }


def _write_common_sources(root: Path) -> None:
    _write_text(root, "AGENTS.md", "Read CODEX.md\n")
    _write_text(root, "CODEX.md", "Spec First\nTests First\n")
    _write_text(root, "CLAUDE.md", "complete: terminal prefix\nblocked_resource\n")
    _write_text(root, "scripts/experiment_template.py", "inference_substrate\n")
    _write_text(root, "scripts/conductor_gates.py", "write_blocked_artifact\n")
    _write_text(root, "research-references.md", "dual-threshold trigger pattern\n")
    _write_text(
        root,
        "openspec/capabilities/verification/spec.md",
        "REQ-VERIFY-3192\nSCENARIO-VERIFY-3192\n"
        "results/experiment_3192_receipt_adversarial_contract_v4.json\n",
    )
    _write_json(
        root,
        mod.EXP3178_REL_PATH,
        {
            "artifact": "experiment_3178_receipt_backed_authenticity_contract_v3",
            "receipt_backed_authenticity_contract_v3_ready": True,
            "required_receipt_fields": [
                "selected_model_id",
                "model_path",
                "model_file_hash",
                "loader_name",
                "substrate_used",
                "prompt_hashes",
                "transcript_hashes",
                "token_counts",
                "random_seed",
                "wall_clock_s",
                "command_hash",
                "subprocess_return_code",
                "stderr_tail",
                "throughput_plausibility",
                "replay_count",
            ],
            "clean_rerun_unlock_requirements": [
                "exp3179.substrate_classification=full_local_sota_receipt",
                "throughput_plausibility.passed=true",
            ],
            "substrate_classification_policy": {
                "classes": {name: {} for name in mod.KNOWN_SUBSTRATE_CLASSES}
            },
            "flagged_adversarial": True,
            "honest_verdict": "complete: v3 ready",
        },
    )
    receipts = [_receipt(0), _receipt(1)]
    _write_json(
        root,
        mod.EXP3179_REL_PATH,
        {
            "artifact": "experiment_3179_local_sota_receipt_smoke_v3",
            "local_sota_receipt_smoke_v3_ready": True,
            "preflight_passed": True,
            "substrate_classification": "cpu_fallback_receipt_only",
            "cpu_fallback_used": True,
            "live_call_count": 2,
            "proof_receipts": receipts,
            "prompt_hashes": [row["prompt_hash"] for row in receipts],
            "transcript_hashes": [row["transcript_hash"] for row in receipts],
            "token_counts": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            "throughput_plausibility_passed": True,
            "throughput_plausibility": {"passed": True},
            "stale_transcript_rejection_passed": True,
            "headline_claim_allowed": False,
            "clean_rerun_allowed": False,
            "selected_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "honest_verdict": "complete: cpu fallback proof only",
        },
    )
    _write_json(
        root,
        mod.EXP3189_REL_PATH,
        {
            "artifact": "experiment_3189_cross_corpus_matrix_v29",
            "cross_corpus_matrix_v29_ready": True,
            "flagged_rows": 18,
            "gated_skip_rows": 14,
            "diagnostic_only_rows": 9,
            "publication_blocker_count": 80,
            "verifier_status": (
                "gated_skip_cpu_fallback_receipt_only_flagged_adversarial_"
                "controlled_invariance_passed_exact_authority_only"
            ),
            "next_top_gap": "full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock",
            "rows": [
                {"row_id": "dot295:exp3179_sota_receipt_smoke", "status": "flagged"},
                {"row_id": "dot295:exp3181_clean_live_rerun_v10", "status": "gated_skipped"},
                {"row_id": "dot295:exp3182_distributional_sidecar", "status": "diagnostic_only"},
            ],
            "honest_verdict": "complete: matrix v29 ready",
        },
    )


def test_req_verify_3192_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3192: OpenSpec declares the v4 contract before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3192" in spec
    assert "SCENARIO-VERIFY-3192" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3192_builds_dual_threshold_contract(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3192: CPU receipts prove execution but not clean reruns."""

    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=12.5,
        tests_run=["SCENARIO-VERIFY-3192 focused"],
    )
    unlocks = artifact["downstream_unlock_fields"]

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3192"
    assert artifact["contract_version"] == "v4"
    assert artifact["receipt_adversarial_contract_v4_ready"] is True
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")

    assert "proof_receipts[].transcript_hash" in artifact["proof_execution_required_fields"]
    assert "proof_receipts[].model_file_hash" in artifact["proof_execution_required_fields"]
    assert "substrate_classification=full_local_sota_receipt" in artifact[
        "clean_rerun_required_fields"
    ]
    assert "cuda_probe.cuda_available=true" in artifact["clean_rerun_required_fields"]

    assert artifact["accepted_substrate_classes"] == [
        "cpu_fallback_receipt_only",
        "full_local_sota_receipt",
    ]
    assert "cpu_fallback_receipt_only" in artifact["rejected_headline_substrate_classes"]
    assert "full_local_sota_receipt" not in artifact["rejected_headline_substrate_classes"]
    assert set(artifact["known_substrate_classes"]) == set(mod.KNOWN_SUBSTRATE_CLASSES)

    assert "methodology_inherited_from_upstream" in artifact["aggregate_required_fields"]
    assert "source_checksums" in artifact["aggregate_required_fields"]
    assert "gate_reasons" in artifact["gated_skip_required_fields"]
    assert "live_call_count=0" in artifact["gated_skip_required_fields"]
    assert "model_specs" in artifact["live_required_fields"]
    assert "preconditions_checked" in artifact["live_required_fields"]
    assert "diagnostic_scope" in artifact["diagnostic_only_required_fields"]
    assert "headline_claim_allowed=false" in artifact["diagnostic_only_required_fields"]

    assert artifact["terminal_verdict_prefixes"] == list(mod.TERMINAL_VERDICT_PREFIXES)
    assert artifact["blocked_verdict_prefixes"] == list(mod.BLOCKED_VERDICT_PREFIXES)
    assert "blocked_receipt_precondition:" in artifact["blocked_verdict_allowances"]

    assert unlocks["proof_execution_sufficient"]["current_value"] is True
    assert unlocks["proof_execution_sufficient"]["allows_cpu_fallback"] is True
    assert unlocks["clean_rerun_allowed"]["current_value"] is False
    assert unlocks["clean_rerun_allowed"]["requires_substrate_class"] == "full_local_sota_receipt"
    assert unlocks["headline_claim_allowed"]["current_value"] is False
    assert unlocks["aggregate_methodology_clean"]["requires_inference_substrate"] == (
        "aggregation_from_upstream_artifacts"
    )

    comparison = artifact["comparison_findings"]
    assert comparison["v3_contract_ready"] is True
    assert comparison["receipt_smoke_substrate_classification"] == "cpu_fallback_receipt_only"
    assert comparison["matrix_v29_publication_blocker_count"] == 80
    assert comparison["prompt_v3_alias_present"] is False

    assessment = artifact["current_evidence_assessment"]
    assert assessment["proof_execution_sufficient"] is True
    assert assessment["clean_rerun_allowed"] is False
    assert assessment["why_clean_rerun_blocked"] == "current_substrate_is_cpu_fallback_receipt_only"
    assert artifact["protected_files_modified_by_this_task"]["scripts/research_conductor.py"] is False


def test_req_verify_3192_writer_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3192: writer persists JSON and validation rejects ambiguous contracts."""

    _write_common_sources(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        started_s=1.0,
        now_s=4.0,
        tests_run=["focused"],
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["duration_s"] == pytest.approx(3.0)
    assert saved["tests_run"] == ["focused"]
    assert saved["source_checksums"][mod.EXP3179_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP3179_REL_PATH
    )

    broken = dict(saved)
    del broken["downstream_unlock_fields"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["accepted_substrate_classes"] = ["full_local_sota_receipt"]
    with pytest.raises(ValueError, match="CPU fallback"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["accepted_substrate_classes"] = [
        "cpu_fallback_receipt_only",
        "full_local_sota_receipt",
        "extra_unreviewed_substrate",
    ]
    with pytest.raises(ValueError, match="accepted substrate classes"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["rejected_headline_substrate_classes"] = ["model_cache_missing"]
    with pytest.raises(ValueError, match="rejected headline"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["rejected_headline_substrate_classes"] = [
        *saved["rejected_headline_substrate_classes"],
        "full_local_sota_receipt",
    ]
    with pytest.raises(ValueError, match="must not be rejected"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["clean_rerun_required_fields"] = ["clean_rerun_allowed"]
    with pytest.raises(ValueError, match="full local SOTA"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["clean_rerun_required_fields"] = ["substrate_classification=full_local_sota_receipt"]
    with pytest.raises(ValueError, match="CUDA"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["downstream_unlock_fields"] = {
        **saved["downstream_unlock_fields"],
        "clean_rerun_allowed": {
            **saved["downstream_unlock_fields"]["clean_rerun_allowed"],
            "current_value": True,
        },
    }
    with pytest.raises(ValueError, match="CPU fallback cannot unlock"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["honest_verdict"] = "success: wrong prefix for this artifact"
    with pytest.raises(ValueError, match="complete"):
        mod.validate_artifact(broken)


def test_req_verify_3192_missing_or_full_substrate_assessment(tmp_path: Path) -> None:
    """REQ-VERIFY-3192: assessment distinguishes missing proof from full SOTA proof."""

    _write_common_sources(tmp_path)
    smoke_path = tmp_path / mod.EXP3179_REL_PATH
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))

    smoke["proof_receipts"][0].pop("model_file_hash")
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    missing_field = mod.build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    assert missing_field["current_evidence_assessment"]["proof_execution_sufficient"] is False
    assert missing_field["current_evidence_assessment"]["proof_missing_fields"] == [
        "proof_receipts[].model_file_hash"
    ]
    assert missing_field["honest_verdict"].startswith("complete:")

    smoke["proof_receipts"][0]["model_file_hash"] = "model-hash-0"
    smoke["substrate_classification"] = "full_local_sota_receipt"
    smoke["cpu_fallback_used"] = False
    smoke["clean_rerun_allowed"] = True
    smoke["headline_claim_allowed"] = True
    smoke["cuda_probe"] = {"cuda_available": True}
    smoke["proof_receipts"][0]["substrate_used"] = "full_local_sota_receipt"
    smoke["proof_receipts"][1]["substrate_used"] = "full_local_sota_receipt"
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    full = mod.build_artifact(tmp_path, started_s=0.0, now_s=1.0)

    assert full["current_evidence_assessment"]["proof_execution_sufficient"] is True
    assert full["current_evidence_assessment"]["clean_rerun_allowed"] is True
    assert full["downstream_unlock_fields"]["clean_rerun_allowed"]["current_value"] is True
    missing_branch = mod.current_evidence_assessment(
        {
            "local_sota_receipt_smoke_v3_ready": True,
            "preflight_passed": True,
            "substrate_classification": "full_local_sota_receipt",
            "live_call_count": 2,
            "proof_receipts": [{}],
            "prompt_hashes": ["p"],
            "transcript_hashes": ["t"],
            "token_counts": {"total_tokens": 1},
            "throughput_plausibility": {"passed": True},
        }
    )
    assert missing_branch["why_clean_rerun_blocked"] == (
        "proof_execution_required_fields_missing"
    )
    unhealthy_branch = mod.current_evidence_assessment(
        {
            "local_sota_receipt_smoke_v3_ready": True,
            "preflight_passed": True,
            "substrate_classification": "cuda_available_unhealthy",
            "live_call_count": 2,
            "proof_receipts": [_receipt(0), _receipt(1)],
            "prompt_hashes": ["p0", "p1"],
            "transcript_hashes": ["t0", "t1"],
            "token_counts": {"total_tokens": 2},
            "throughput_plausibility": {"passed": True},
        }
    )
    assert unhealthy_branch["why_clean_rerun_blocked"] == "full_local_sota_receipt_not_established"
    assert mod.proof_missing_fields({}, [])[:3] == [
        "local_sota_receipt_smoke_v3_ready",
        "substrate_classification",
        "live_call_count",
    ]
    assert mod.throughput_passed({"throughput_plausibility": {"passed": True}}) is True
    assert mod.mapping_list("not-a-list") == []
    assert mod.int_or_zero("not-an-int") == 0
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad json}\n", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    assert mod.duration(5.0, 4.0) == 0.0
