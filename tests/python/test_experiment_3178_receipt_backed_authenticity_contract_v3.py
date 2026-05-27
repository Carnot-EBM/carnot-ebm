"""Tests for Exp 3178 receipt-backed authenticity contract v3.

Spec refs: REQ-VERIFY-3178, SCENARIO-VERIFY-3178.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import receipt_backed_authenticity_contract_v3 as mod


REQUIRED_FIELDS = {
    "receipt_backed_authenticity_contract_v3_ready",
    "inherited_v2_contract_fields",
    "substrate_classification_policy",
    "required_receipt_fields",
    "cpu_fallback_policy",
    "fake_evidence_rejection_criteria",
    "clean_rerun_unlock_requirements",
    "headline_claim_policy",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}

SUBSTRATE_CLASSES = {
    "model_cache_missing",
    "loader_missing",
    "cuda_unavailable",
    "cuda_available_unhealthy",
    "cpu_fallback_receipt_only",
    "full_local_sota_receipt",
}

RECEIPT_FIELDS = {
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


def _exp3164_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3164_duration_corrected_authenticity_contract_v2",
        "duration_corrected_authenticity_contract_v2_ready": True,
        "measured_work_requirements": [
            {
                "requirement": "local_model_path_exists",
                "source_field": "model_load_evidence.path_exists",
                "observed": True,
            },
            {
                "requirement": "model_load_proof",
                "source_field": "model_load_evidence.load_wall_time_s",
                "observed": True,
            },
            {
                "requirement": "transcript_and_prompt_hashes",
                "source_field": "transcript_hashes",
                "observed": True,
            },
        ],
        "required_preflight_fields": [
            "model_specs.path_exists",
            "model_load_evidence.load_attempted",
            "model_load_evidence.returncode",
            "transcript_hashes.transcript_sha256",
            "prompt_hashes",
            "token_counts.total_tokens",
            "random_seed",
            "reproducibility_checksum",
            "controlled_subprocess_return_codes",
            "inference_substrate",
        ],
        "fake_evidence_rejection_criteria": [
            "reject no model loaded",
            "reject missing transcript hashes",
            "reject impossible token throughput",
        ],
        "headline_claim_policy": {
            "smoke_test_headline_claim_allowed": False,
            "smoke_test_role": "preflight authenticity evidence only",
        },
        "honest_verdict": "complete: duration_corrected_authenticity_contract_v2_ready=true",
    }


def _exp3165_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3165_live_sota_authenticity_replay_v2",
        "live_sota_authenticity_replay_v2_ready": True,
        "preflight_passed": False,
        "blocked_reason": "CUDA/GPU substrate unavailable for mandated GGUF replay smoke",
        "locally_usable_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
        "live_call_count": 0,
        "model_load_evidence": {
            "load_attempted": False,
            "runtime": "llama_cpp",
            "selected_model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "selected_model_path": "/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
            "path_exists": True,
            "returncode": None,
            "stderr_summary": "",
        },
        "inference_substrate": {
            "kind": "live_sota_authenticity_replay_v2",
            "runtime": "llama_cpp",
            "executes_models": False,
            "live_model_calls": 0,
            "legacy_small_model_used": False,
            "gpu_probe": {
                "cuda_available": False,
                "gpu_count": 1,
                "torch_cuda_probe": {
                    "returncode": 0,
                    "stderr_summary": "CUDA initialization: Error 101 invalid device ordinal",
                },
                "nvidia_smi_inventory": {"returncode": 0, "available": True},
            },
        },
        "honest_verdict": "blocked_gpu_substrate: preflight_passed=false",
    }


def _write_common_sources(root: Path) -> None:
    _write_text(root, "AGENTS.md", "Read CODEX.md\n")
    _write_text(root, "CODEX.md", "Spec First\nTests First\n")
    _write_text(root, "CLAUDE.md", "All headline results must have live GPU provenance.\n")
    _write_text(
        root,
        "scripts/experiment_template.py",
        "from carnot.inference.sota_models import cached_sota_pair\n",
    )
    _write_text(
        root,
        "research-references.md",
        "Receipt-backed local SOTA execution before new headline claims.\n",
    )
    _write_text(
        root,
        "openspec/capabilities/verification/spec.md",
        "REQ-VERIFY-3178\nSCENARIO-VERIFY-3178\n"
        "results/experiment_3178_receipt_backed_authenticity_contract_v3.json\n",
    )
    _write_json(root, mod.EXP3164_REL_PATH, _exp3164_payload())
    _write_json(root, mod.EXP3165_REL_PATH, _exp3165_payload())
    _write_json(
        root,
        mod.EXP3167_REL_PATH,
        {
            "artifact": "experiment_3167_clean_live_sota_verifier_rerun_v9",
            "clean_live_verifier_rerun_v9_ready": True,
            "gated_skip": True,
            "gated_skip_reason": "exp3165 preflight_passed=false; clean rerun cannot call a model",
            "controlled_invariance_passed": False,
            "headline_claim_allowed": False,
            "honest_verdict": "complete: clean_live_verifier_rerun_v9_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3176_REL_PATH,
        {
            "artifact": "experiment_3176_capstone_v294",
            "capstone_v294_ready": True,
            "paper_ready": False,
            "next_top_gap": "receipt_backed_local_sota_execution",
            "honest_verdict": "complete: capstone_v294_ready=true",
        },
    )


def test_req_verify_3178_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3178: OpenSpec declares the v3 contract before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3178" in spec
    assert "SCENARIO-VERIFY-3178" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "model_cache_missing" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3178_builds_v3_contract(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3178: v3 separates substrate blockers and headline policy."""

    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=12.0,
        tests_run=["REQ-VERIFY-3178 focused"],
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["receipt_backed_authenticity_contract_v3_ready"] is True
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["exp3165_blocker_reason"] == (
        "CUDA/GPU substrate unavailable for mandated GGUF replay smoke"
    )
    assert artifact["exp3165_substrate_observation"]["classified_as"] == "cuda_unavailable"

    inherited = artifact["inherited_v2_contract_fields"]
    assert "model_specs.path_exists" in inherited
    assert "controlled_subprocess_return_codes" in inherited
    measured = artifact["v2_measured_work_requirements"]
    assert [row["requirement"] for row in measured] == [
        "local_model_path_exists",
        "model_load_proof",
        "transcript_and_prompt_hashes",
    ]

    policy = artifact["substrate_classification_policy"]
    assert set(policy["classes"]) == SUBSTRATE_CLASSES
    assert policy["classes"]["model_cache_missing"]["blocks_live_model_call"] is True
    assert policy["classes"]["loader_missing"]["blocks_live_model_call"] is True
    assert policy["classes"]["cuda_unavailable"]["cpu_fallback_class"] == (
        "cpu_fallback_receipt_only"
    )
    assert policy["classes"]["full_local_sota_receipt"]["headline_eligible"] is True

    assert set(artifact["required_receipt_fields"]) == RECEIPT_FIELDS
    assert artifact["cpu_fallback_policy"]["admissible_for_receipt_wiring"] is True
    assert artifact["cpu_fallback_policy"]["headline_verifier_benchmark_allowed"] is False
    assert artifact["cpu_fallback_policy"]["clean_rerun_unlock_allowed"] is False
    assert artifact["headline_claim_policy"]["one_prompt_smoke_headline_allowed"] is False
    assert artifact["headline_claim_policy"]["cpu_fallback_headline_allowed"] is False
    assert "controlled_invariance_passed=true" in artifact["clean_rerun_unlock_requirements"]
    assert "exact_authority_scoring_passed=true" in artifact["clean_rerun_unlock_requirements"]

    criteria = "\n".join(artifact["fake_evidence_rejection_criteria"])
    assert "CPU-only evidence promoted as headline" in criteria
    assert "one-prompt smoke promoted as benchmark" in criteria
    assert "missing CUDA health evidence for headline" in criteria
    assert artifact["inference_substrate"]["executes_models"] is False
    assert artifact["inference_substrate"]["live_model_calls"] == 0


def test_req_verify_3178_writer_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3178: writer persists JSON and validation rejects overclaims."""

    _write_common_sources(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        started_s=1.0,
        now_s=3.5,
        tests_run=["focused"],
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["receipt_backed_authenticity_contract_v3_ready"] is True
    assert saved["duration_s"] == pytest.approx(2.5)
    assert saved["source_checksums"][mod.EXP3165_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP3165_REL_PATH
    )

    broken = dict(saved)
    broken["substrate_classification_policy"] = {
        **saved["substrate_classification_policy"],
        "classes": {"full_local_sota_receipt": {}},
    }
    with pytest.raises(ValueError, match="substrate classes"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["required_receipt_fields"] = ["selected_model_id"]
    with pytest.raises(ValueError, match="required receipt fields"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["cpu_fallback_policy"] = {
        **saved["cpu_fallback_policy"],
        "headline_verifier_benchmark_allowed": True,
    }
    with pytest.raises(ValueError, match="CPU fallback"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["cpu_fallback_policy"] = {
        **saved["cpu_fallback_policy"],
        "clean_rerun_unlock_allowed": True,
    }
    with pytest.raises(ValueError, match="clean rerun"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["inference_substrate"] = {**saved["inference_substrate"], "executes_models": True}
    with pytest.raises(ValueError, match="no live model"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    del broken["headline_claim_policy"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["contract_blockers"] = ["should_not_be_ready"]
    with pytest.raises(ValueError, match="ready contract"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["honest_verdict"] = "maybe later"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(broken)


def test_req_verify_3178_blocks_when_v2_contract_missing(tmp_path: Path) -> None:
    """REQ-VERIFY-3178: missing v2 authority is an honest contract precondition block."""

    _write_common_sources(tmp_path)
    for rel_path in (
        mod.EXP3164_REL_PATH,
        mod.EXP3165_REL_PATH,
        mod.EXP3167_REL_PATH,
        mod.EXP3176_REL_PATH,
    ):
        (tmp_path / rel_path).unlink()

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=3.0)

    assert artifact["receipt_backed_authenticity_contract_v3_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_missing_v2_contract:")
    assert artifact["contract_blockers"] == [
        "missing_exp3164_v2_contract",
        "missing_exp3165_replay_artifact",
        "missing_exp3167_clean_rerun_artifact",
        "missing_exp3176_capstone_artifact",
    ]
    assert artifact["inherited_v2_contract_fields"] == list(mod.DEFAULT_INHERITED_V2_FIELDS)
    assert artifact["exp3165_blocker_reason"] == "missing_exp3165_blocked_reason"
    assert artifact["exp3165_substrate_observation"]["classified_as"] == "model_cache_missing"


def test_req_verify_3178_substrate_classifier_edges() -> None:
    """REQ-VERIFY-3178: all v3 substrate classes are explicitly reachable."""

    assert (
        mod.classify_substrate(
            {"locally_usable_model_ids": []},
            "no cache",
            {"cuda_available": True},
            {"path_exists": False},
        )
        == "model_cache_missing"
    )
    assert (
        mod.classify_substrate(
            {"locally_usable_model_ids": ["model"]},
            "loader missing for llama_cpp",
            {"cuda_available": True},
            {"path_exists": True},
        )
        == "loader_missing"
    )
    assert (
        mod.classify_substrate(
            {"locally_usable_model_ids": ["model"], "preflight_passed": True, "live_call_count": 2},
            "complete",
            {"cuda_available": True},
            {"path_exists": True},
        )
        == "full_local_sota_receipt"
    )
    assert (
        mod.classify_substrate(
            {"locally_usable_model_ids": ["model"], "preflight_passed": False},
            "cpu fallback receipt",
            {"cuda_available": True},
            {"path_exists": True},
        )
        == "cpu_fallback_receipt_only"
    )
    assert (
        mod.classify_substrate(
            {"locally_usable_model_ids": ["model"], "preflight_passed": False},
            "cuda load failed",
            {"cuda_available": True},
            {"path_exists": True},
        )
        == "cuda_available_unhealthy"
    )
    assert mod.int_or_zero("not-an-int") == 0
