"""Tests for Exp 3164 duration-corrected authenticity contract v2.

Spec refs: REQ-VERIFY-3164, SCENARIO-VERIFY-3164.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import duration_corrected_authenticity_contract_v2 as mod


MODEL_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
MODEL_PATH = "/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"

REQUIRED_FIELDS = {
    "duration_corrected_authenticity_contract_v2_ready",
    "old_fixed_duration_rule_retired_as_hard_gate",
    "measured_work_requirements",
    "token_scaled_duration_policy",
    "repeated_call_policy",
    "required_preflight_fields",
    "fake_evidence_rejection_criteria",
    "headline_claim_policy",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exp3151_payload(
    *,
    transcript_hashes: list[dict[str, Any]] | None = None,
    random_seed: int | None = 20260526,
    reproducibility_checksum: str | None = "abc123checksum",
    selected_model_id: str = MODEL_ID,
    selected_model_path: str = MODEL_PATH,
    returncode: int | None = 0,
    path_exists: bool = True,
    duration_s: float = 10.590346,
) -> dict[str, Any]:
    transcripts = (
        transcript_hashes
        if transcript_hashes is not None
        else [
            {
                "model_id": MODEL_ID,
                "prompt_hash": "prompt-hash-1",
                "response_hash": "response-hash-1",
                "transcript_sha256": "transcript-hash-1",
                "prompt_token_count": 19,
                "output_token_count": 5,
                "random_seed": 20260526,
            }
        ]
    )
    worker_stdout = {
        "ok": True,
        "runtime": "llama_cpp",
        "load_wall_time_s": 7.96673,
        "generation_wall_time_s": 0.432167,
        "total_worker_wall_time_s": 8.546801,
        "output_text": "VALID",
        "usage": {
            "prompt_tokens": 19,
            "completion_tokens": 5,
            "total_tokens": 24,
        },
    }
    payload: dict[str, Any] = {
        "artifact": "experiment_3151_live_inference_authenticity_preflight_v1",
        "schema": "carnot.live_inference_authenticity_preflight.v1",
        "run_date": "20260526",
        "duration_s": duration_s,
        "blocked_reason": (
            f"duration_s={duration_s} is shorter than minimum plausible duration 60.0"
        ),
        "minimum_duration_requirement_s": 60.0,
        "preflight_passed": False,
        "live_call_count": len(transcripts),
        "selected_model_ids": [selected_model_id],
        "locally_usable_model_ids": [MODEL_ID],
        "model_load_evidence": {
            "load_attempted": True,
            "runtime": "llama_cpp",
            "selected_model_id": selected_model_id,
            "selected_model_path": selected_model_path,
            "path_exists": path_exists,
            "load_command": ["/repo/.venv/bin/python", "-c", "worker"],
            "load_command_sha256": "command-hash-1",
            "worker_code_sha256": "worker-hash-1",
            "returncode": returncode,
            "load_wall_time_s": 7.96673,
            "generation_wall_time_s": 0.432167,
            "total_worker_wall_time_s": 8.546801,
            "stdout_summary": json.dumps(worker_stdout, sort_keys=True) + "\n",
            "stderr_summary": "llama_context: fixture\n",
            "runtime_error": None,
        },
        "model_specs": [
            {
                "hf_id": MODEL_ID,
                "model_path": selected_model_path,
                "path_exists": path_exists,
                "usable_locally": path_exists,
                "selected_for_smoke": True,
            }
        ],
        "token_counts": {
            "prompt_tokens": 19,
            "completion_tokens": 5,
            "total_tokens": 24,
            "source": "llama_cpp_usage",
        },
        "transcript_hashes": transcripts,
        "random_seed": random_seed,
        "reproducibility_checksum": reproducibility_checksum,
        "inference_substrate": {
            "kind": "live_inference_authenticity_preflight_v1",
            "runtime": "llama_cpp",
            "executes_models": bool(transcripts),
            "live_model_calls": len(transcripts),
            "model_load_attempted": True,
            "selected_model_id": selected_model_id,
            "selected_model_path": selected_model_path,
            "gpu_probe": {
                "cuda_available": True,
                "gpu_count": 2,
                "torch_cuda_probe": {"returncode": 0},
                "nvidia_smi_inventory": {"returncode": 0},
            },
            "cpu_probe": {"platform": "Linux", "python_version": "3.14.4"},
        },
        "honest_verdict": "blocked_duration_too_short: preflight_passed=false",
    }
    return payload


def _write_common_sources(root: Path, *, exp3151: dict[str, Any] | None = None) -> None:
    _write_text(root, "AGENTS.md", "Read CODEX.md\n")
    _write_text(root, "CODEX.md", "Spec First\nTests First\n")
    _write_text(root, "CLAUDE.md", "All headline results must have live GPU provenance.\n")
    _write_text(root, "research-references.md", "Post-.293 planning sweep\n")
    _write_text(root, "scripts/experiment_template.py", "DEFAULT_BATCH_SIZE = 8\n")
    _write_text(
        root,
        "openspec/capabilities/verification/spec.md",
        "REQ-VERIFY-3164\nSCENARIO-VERIFY-3164\n"
        "results/experiment_3164_duration_corrected_authenticity_contract_v2.json\n",
    )
    _write_json(
        root,
        mod.EXP3150_REL_PATH,
        {
            "artifact": "experiment_3150_adversarial_verifier_evidence_corrigendum_v1",
            "methodology_requirements_for_rerun": [
                "record random_seed or random_seeds_used for every live row",
                "record reproducibility_checksum over prompts, raw outputs, model specs, and runner revision",
                "record transcript path and sha256 for every live call",
            ],
            "honest_verdict": "complete: adversarial_corrigendum_v1_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3151_REL_PATH,
        exp3151 if exp3151 is not None else _exp3151_payload(),
    )
    _write_json(
        root,
        mod.EXP3162_REL_PATH,
        {
            "artifact": "experiment_3162_capstone_v293",
            "next_top_gap": "clean_live_verifier_corrigendum_repair_gate",
            "publication_blocker_count": 65,
            "missing_artifacts": [
                {
                    "experiment_id": "exp3152",
                    "path": "results/experiment_3152_clean_live_sota_verifier_rerun_v8.json",
                }
            ],
            "honest_verdict": "complete: capstone_v293_ready=true",
        },
    )


def test_req_verify_3164_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3164: OpenSpec declares the v2 contract before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3164" in spec
    assert "SCENARIO-VERIFY-3164" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "old fixed 60-second duration rule" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3164_builds_duration_corrected_contract(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3164: fast smoke is governed by measured-work evidence."""

    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=12.5,
        tests_run=["REQ-VERIFY-3164 focused"],
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["duration_corrected_authenticity_contract_v2_ready"] is True
    assert artifact["old_fixed_duration_rule_retired_as_hard_gate"] is True
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["run_date"] == "20260527"
    assert artifact["honest_verdict"].startswith("complete:")

    extracted = artifact["exp3151_extracted_measurements"]
    assert extracted["model_load_wall_time_s"] == pytest.approx(7.96673)
    assert extracted["generation_wall_time_s"] == pytest.approx(0.432167)
    assert extracted["token_counts"] == {
        "prompt_tokens": 19,
        "completion_tokens": 5,
        "total_tokens": 24,
        "source": "llama_cpp_usage",
    }
    assert extracted["selected_model_id"] == MODEL_ID
    assert extracted["selected_model_path"] == MODEL_PATH
    assert extracted["load_command_sha256"] == "command-hash-1"
    assert extracted["worker_code_sha256"] == "worker-hash-1"
    assert extracted["transcript_hash_available"] is True
    assert extracted["prompt_hashes"] == ["prompt-hash-1"]
    assert extracted["controlled_subprocess_return_codes"] == [
        {"name": "model_load_smoke_worker", "returncode": 0},
        {"name": "torch_cuda_probe", "returncode": 0},
        {"name": "nvidia_smi_inventory", "returncode": 0},
    ]
    assert extracted["wall_clock_supported_by_command_output"] is True
    assert extracted["substrate_evidence"]["live_model_calls"] == 1

    policy = artifact["token_scaled_duration_policy"]
    assert policy["fixed_60s_floor"]["hard_gate"] is False
    assert policy["fixed_60s_floor"]["optional_warning_only_for_large_panels"] is True
    assert policy["one_prompt_smoke"]["fixed_minimum_duration_s"] == pytest.approx(0.0)
    assert policy["observed_exp3151_smoke"]["blocked_by_old_fixed_floor"] is True
    assert policy["observed_exp3151_smoke"]["accepted_by_v2_duration_policy"] is True

    repeated = artifact["repeated_call_policy"]
    assert repeated["exp3165"]["minimum_distinct_smoke_calls"] == 2
    assert repeated["exp3167"]["minimum_distinct_smoke_calls"] == 3
    assert repeated["stale_replay_controls"]["reject_reused_transcript_sha256"] is True

    criteria = "\n".join(artifact["fake_evidence_rejection_criteria"])
    assert "no model loaded" in criteria
    assert "missing transcript hashes" in criteria
    assert "reused stale transcript hash" in criteria
    assert "wall-clock claims not supported by command output" in criteria

    fields = artifact["required_preflight_fields"]
    assert "model_load_evidence.load_command_sha256" in fields
    assert "transcript_hashes.transcript_sha256" in fields
    assert "prompt_hashes" in fields
    assert "random_seed" in fields
    assert "reproducibility_checksum" in fields
    assert "controlled_subprocess_return_codes" in fields

    assert artifact["headline_claim_policy"]["smoke_test_headline_claim_allowed"] is False
    assert set(artifact["reusable_contracts"]) == {"exp3165", "exp3167"}
    assert artifact["source_checksums"][mod.EXP3151_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP3151_REL_PATH
    )
    assert artifact["inference_substrate"]["no_live_llm_inference"] is True
    assert artifact["inference_substrate"]["live_model_calls_run_by_exp3164"] == 0


def test_req_verify_3164_blocks_on_missing_machine_checkable_source_evidence(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3164: missing transcript, seed, checksum, and load proof fail closed."""

    _write_common_sources(
        tmp_path,
        exp3151=_exp3151_payload(
            transcript_hashes=[],
            random_seed=None,
            reproducibility_checksum=None,
            returncode=1,
            path_exists=False,
        ),
    )

    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=4.0)

    assert artifact["duration_corrected_authenticity_contract_v2_ready"] is False
    assert artifact["duration_s"] == pytest.approx(0.0)
    assert artifact["honest_verdict"].startswith("blocked_contract_source_evidence:")
    assert artifact["exp3151_extracted_measurements"]["transcript_hash_available"] is False
    assert artifact["observed_source_assessment"]["passed"] is False
    assert artifact["observed_source_assessment"]["violations"] == [
        "model path existence proof missing",
        "controlled model-load subprocess did not return 0",
        "missing transcript hashes",
        "missing prompt hashes",
        "missing random_seed",
        "missing reproducibility_checksum",
    ]
    assert artifact["reusable_contracts"]["exp3165"]["usable_by_downstream"] is False


def test_req_verify_3164_write_artifact_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3164: writer persists JSON and validation rejects overclaims."""

    _write_common_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        started_s=2.0,
        now_s=4.25,
        tests_run=["focused"],
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["duration_corrected_authenticity_contract_v2_ready"] is True
    assert saved["tests_run"] == ["focused"]
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad json}\n", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.stable_hash({"b": 2, "a": 1}) == mod.stable_hash({"a": 1, "b": 2})
    assert mod.duration(9.0, 3.0) == 0.0
    assert mod.parse_worker_stdout_summary("bad\n{\"ok\": true}\n") == {"ok": True}
    assert mod.parse_worker_stdout_summary("no-json\n") == {}
    assert mod.model_path_matches_selected_spec("missing", MODEL_PATH, []) is False
    assert mod.int_or_none("not-int") is None
    assert mod.float_or_none("not-float") is None
    assert mod._first([]) is None

    bad_assessment = mod.assess_observed_source(
        {
            "model_path_exists_proof": True,
            "model_load_wall_time_s": None,
            "model_load_returncode": 0,
            "transcript_hash_available": True,
            "prompt_hashes": ["prompt"],
            "token_counts": {},
            "random_seed": 1,
            "reproducibility_checksum": "checksum",
            "completion_tokens_per_generation_second": 501.0,
            "model_path_matches_selected_spec": False,
            "wall_clock_supported_by_command_output": False,
            "blocked_by_old_fixed_floor": True,
        }
    )
    assert bad_assessment["violations"] == [
        "model-load wall time missing",
        "missing token counts",
        "impossible token throughput",
        "selected model/local path mismatch",
        "wall-clock claims not supported by command output",
    ]

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="old fixed duration"):
        mod.validate_artifact(saved | {"old_fixed_duration_rule_retired_as_hard_gate": False})
    with pytest.raises(ValueError, match="headline"):
        mod.validate_artifact(
            saved | {"headline_claim_policy": {"smoke_test_headline_claim_allowed": True}}
        )
    with pytest.raises(ValueError, match="no live model inference"):
        mod.validate_artifact(
            saved
            | {
                "inference_substrate": {
                    "no_live_llm_inference": False,
                    "live_model_calls_run_by_exp3164": 1,
                }
            }
        )
    with pytest.raises(ValueError, match="terminal success prefix"):
        mod.validate_artifact(saved | {"honest_verdict": "blocked_wrong:"})
    with pytest.raises(ValueError, match="blocked_"):
        mod.validate_artifact(
            saved
            | {
                "duration_corrected_authenticity_contract_v2_ready": False,
                "honest_verdict": "complete: wrong",
            }
        )
