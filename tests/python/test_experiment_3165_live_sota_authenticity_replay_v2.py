"""Tests for Exp 3165 live SOTA authenticity replay v2.

Spec refs: REQ-VERIFY-3165, SCENARIO-VERIFY-3165.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import live_sota_authenticity_replay_v2 as mod


MODEL_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "source\n") -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _model_path(cache_root: Path) -> Path:
    return (
        cache_root
        / "models--unsloth--gemma-4-26B-A4B-it-GGUF"
        / "snapshots"
        / "abc123"
        / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    )


def _write_common_sources(
    root: Path,
    *,
    exp3164_ready: bool = True,
    model_path: Path | None = None,
) -> None:
    _write_text(root, "AGENTS.md", "Read CODEX.md\n")
    _write_text(root, "CODEX.md", "Spec First\nTests First\n")
    _write_text(root, "CLAUDE.md", "All headline results must have live GPU provenance.\n")
    _write_text(root, "scripts/experiment_template.py", "DEFAULT_BATCH_SIZE = 8\n")
    _write_text(
        root,
        "openspec/capabilities/verification/spec.md",
        "REQ-VERIFY-3165\nSCENARIO-VERIFY-3165\n"
        "results/experiment_3165_live_sota_authenticity_replay_v2.json\n",
    )
    _write_json(
        root,
        mod.EXP3164_REL_PATH,
        {
            "artifact": "experiment_3164_duration_corrected_authenticity_contract_v2",
            "duration_corrected_authenticity_contract_v2_ready": exp3164_ready,
            "reusable_contracts": {
                "exp3165": {
                    "minimum_distinct_smoke_calls": 2,
                    "old_fixed_60s_rule_hard_gate": False,
                    "usable_by_downstream": exp3164_ready,
                }
            },
            "token_scaled_duration_policy": {
                "one_prompt_smoke": {
                    "reject_if_completion_tokens_per_second_gt": 500.0,
                }
            },
            "fake_evidence_rejection_criteria": [
                "reject no model loaded",
                "reject missing transcript hashes",
            ],
            "honest_verdict": "complete: duration_corrected_authenticity_contract_v2_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3151_REL_PATH,
        {"artifact": "experiment_3151_live_inference_authenticity_preflight_v1"},
    )
    inventory_path = str(model_path) if model_path is not None else None
    _write_json(
        root,
        mod.EXP3123_REL_PATH,
        {
            "artifact": "experiment_3123_sota_cache_preconditions_manifest_v2",
            "cache_inventory": [
                {
                    "hf_id": MODEL_ID,
                    "cache_status": "resolved" if inventory_path else "missing",
                    "path": inventory_path,
                    "expected_quantization": "Q4_K_M",
                }
            ],
            "present_model_ids": [MODEL_ID] if inventory_path else [],
        },
    )


def _runner(payload: dict[str, Any] | None = None):
    def run(command: list[str], *, timeout_s: int = 10, env: dict[str, str] | None = None):
        del timeout_s, env
        if command[:2] == ["/venv/python", "-c"] and "torch" in command[2]:
            return {
                "command": command,
                "returncode": 0,
                "stdout": "2.11.0+cu128 True 2\n",
                "stderr": "",
            }
        if command and command[0] == "nvidia-smi":
            return {
                "command": command,
                "returncode": 0,
                "stdout": (
                    "0, NVIDIA GeForce RTX 3090, 24576, 4, 24123, 595.71.05\n"
                    "1, NVIDIA GeForce RTX 3090, 24576, 4, 24123, 595.71.05\n"
                ),
                "stderr": "",
            }
        if "--exp3165-smoke-worker" in command:
            worker_payload = payload or {
                "ok": True,
                "runtime": "llama_cpp",
                "load_wall_time_s": 3.25,
                "total_worker_wall_time_s": 4.75,
                "calls": [
                    {
                        "prompt": mod.DEFAULT_PROMPTS[0],
                        "seed": mod.DEFAULT_RANDOM_SEED,
                        "output_text": "READY",
                        "generation_wall_time_s": 0.5,
                        "usage": {
                            "prompt_tokens": 7,
                            "completion_tokens": 1,
                            "total_tokens": 8,
                        },
                    },
                    {
                        "prompt": mod.DEFAULT_PROMPTS[1],
                        "seed": mod.DEFAULT_RANDOM_SEED + 1,
                        "output_text": "VERIFIED",
                        "generation_wall_time_s": 0.75,
                        "usage": {
                            "prompt_tokens": 7,
                            "completion_tokens": 1,
                            "total_tokens": 8,
                        },
                    },
                ],
            }
            return {
                "command": command,
                "returncode": 0,
                "stdout": json.dumps(worker_payload, sort_keys=True) + "\n",
                "stderr": "llama_context: fixture\n",
            }
        raise AssertionError(f"unexpected command: {command}")

    return run


def test_req_verify_3165_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3165: OpenSpec declares the replay before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3165" in spec
    assert "SCENARIO-VERIFY-3165" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "Exp 3164 contract first" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3165_passes_with_two_distinct_smoke_calls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3165: two fresh smoke calls satisfy the v2 replay gate."""

    cache_root = tmp_path / "hf-cache"
    model_path = _model_path(cache_root)
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"gguf fixture bytes")
    _write_common_sources(tmp_path, model_path=model_path)

    artifact = mod.build_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python="/venv/python",
        command_runner=_runner(),
        started_s=10.0,
        now_s=15.5,
        tests_run=["REQ-VERIFY-3165 focused"],
    )

    assert artifact["live_sota_authenticity_replay_v2_ready"] is True
    assert artifact["preflight_passed"] is True
    assert artifact["live_call_count"] == 2
    assert artifact["measured_work_policy_passed"] is True
    assert artifact["fake_evidence_rejection_passed"] is True
    assert artifact["token_scaled_duration_policy_passed"] is True
    assert artifact["headline_claim_allowed"] is False
    assert artifact["blocked_reason"] == ""
    assert artifact["honest_verdict"].startswith("complete:")

    assert artifact["locally_usable_model_ids"] == [MODEL_ID]
    assert artifact["selected_model_ids"] == [MODEL_ID]
    assert artifact["unavailable_model_ids"] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
    ]
    selected = [row for row in artifact["model_specs"] if row["selected_for_smoke"]]
    assert selected[0]["hf_id"] == MODEL_ID
    assert selected[0]["model_path"] == str(model_path)
    assert selected[0]["path_exists"] is True

    load = artifact["model_load_evidence"]
    assert load["load_attempted"] is True
    assert load["selected_model_path"] == str(model_path)
    assert load["load_wall_time_s"] == pytest.approx(3.25)
    assert load["generation_wall_time_s"] == pytest.approx(1.25)
    assert load["per_call_generation_wall_time_s"] == [0.5, 0.75]
    assert load["load_command_sha256"] == mod.stable_hash(load["load_command"])
    assert load["worker_code_sha256"] == mod.sha256_text(mod.SMOKE_WORKER_CODE)

    assert len(set(artifact["prompt_hashes"])) == 2
    assert len(set(row["transcript_sha256"] for row in artifact["transcript_hashes"])) == 2
    assert [row["random_seed"] for row in artifact["transcript_hashes"]] == [
        mod.DEFAULT_RANDOM_SEED,
        mod.DEFAULT_RANDOM_SEED + 1,
    ]
    assert artifact["token_counts"] == {
        "prompt_tokens": 14,
        "completion_tokens": 2,
        "total_tokens": 16,
        "source": "llama_cpp_usage",
        "per_call": [
            {
                "prompt_tokens": 7,
                "completion_tokens": 1,
                "total_tokens": 8,
                "source": "llama_cpp_usage",
            },
            {
                "prompt_tokens": 7,
                "completion_tokens": 1,
                "total_tokens": 8,
                "source": "llama_cpp_usage",
            },
        ],
    }
    assert artifact["controlled_subprocess_return_codes"] == [
        {"name": "torch_cuda_probe", "returncode": 0},
        {"name": "nvidia_smi_inventory", "returncode": 0},
        {"name": "model_load_smoke_worker", "returncode": 0},
    ]
    assert artifact["inference_substrate"]["downloads_models"] is False
    assert artifact["source_artifacts"][0]["role"] == "exp3164_v2_contract"
    assert artifact["reproducibility_checksum"]


def test_req_verify_3165_blocks_without_mandated_cache_and_does_not_smoke(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3165: absent mandated GGUFs produce a complete blocked artifact."""

    cache_root = tmp_path / "hf-cache"
    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python="/venv/python",
        command_runner=_runner(),
        started_s=1.0,
        now_s=3.0,
    )

    assert artifact["live_sota_authenticity_replay_v2_ready"] is True
    assert artifact["preflight_passed"] is False
    assert artifact["live_call_count"] == 0
    assert artifact["model_load_evidence"]["load_attempted"] is False
    assert artifact["selected_model_ids"] == []
    assert artifact["prompt_hashes"] == []
    assert artifact["transcript_hashes"] == []
    assert artifact["token_counts"]["total_tokens"] == 0
    assert artifact["measured_work_policy_passed"] is False
    assert artifact["fake_evidence_rejection_passed"] is True
    assert artifact["headline_claim_allowed"] is False
    assert artifact["blocked_reason"] == "no mandated local SOTA GGUF path exists with nonzero size"
    assert artifact["honest_verdict"].startswith("blocked_no_mandated_sota_gguf:")


def test_req_verify_3165_rejects_incomplete_repeated_call_evidence(tmp_path: Path) -> None:
    """REQ-VERIFY-3165: one transcript is not enough for the replay contract."""

    cache_root = tmp_path / "hf-cache"
    model_path = _model_path(cache_root)
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"gguf fixture bytes")
    _write_common_sources(tmp_path, model_path=model_path)

    artifact = mod.build_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python="/venv/python",
        command_runner=_runner(
            {
                "ok": True,
                "runtime": "llama_cpp",
                "load_wall_time_s": 3.25,
                "total_worker_wall_time_s": 4.0,
                "calls": [
                    {
                        "prompt": mod.DEFAULT_PROMPTS[0],
                        "seed": mod.DEFAULT_RANDOM_SEED,
                        "output_text": "READY",
                        "generation_wall_time_s": 0.5,
                        "usage": {
                            "prompt_tokens": 7,
                            "completion_tokens": 1,
                            "total_tokens": 8,
                        },
                    }
                ],
            }
        ),
    )

    assert artifact["preflight_passed"] is False
    assert artifact["live_call_count"] == 1
    assert artifact["repeated_call_policy_passed"] is False
    assert artifact["blocked_reason"] == "repeated-call smoke produced 1 transcripts; expected at least 2"
    assert artifact["honest_verdict"].startswith("blocked_repeated_call_policy:")


def test_req_verify_3165_writer_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3165: writer persists JSON and validation rejects overclaims."""

    cache_root = tmp_path / "hf-cache"
    model_path = _model_path(cache_root)
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"gguf fixture bytes")
    _write_common_sources(tmp_path, model_path=model_path)

    output = mod.write_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python="/venv/python",
        command_runner=_runner(),
        started_s=2.0,
        now_s=4.25,
        tests_run=["focused"],
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["tests_run"] == ["focused"]
    assert saved["duration_s"] == pytest.approx(2.25)
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad json}\n", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert str(mod.default_hf_cache_root()).endswith(".cache/huggingface/hub")
    assert mod.direct_cache_candidates(cache_root, MODEL_ID) == [model_path]
    rel = tmp_path / "relative.gguf"
    rel.write_bytes(b"relative")
    assert mod.path_evidence(tmp_path, "relative.gguf")["exists"] is True
    assert mod.path_evidence(tmp_path, "missing.gguf")["exists"] is False
    large = tmp_path / "large.gguf"
    large.write_bytes(b"x" * (1024 * 1024 + 1))
    assert mod.bounded_file_hash(large)
    assert mod.stable_hash({"b": 2, "a": 1}) == mod.stable_hash({"a": 1, "b": 2})
    assert mod.duration(9.0, 3.0) == 0.0
    assert mod.int_or_none("not-int") is None
    assert mod.float_or_none("not-float") is None
    assert mod.first_json_line("bad\n{\"ok\": true}\n") == {"ok": True}
    assert mod.first_json_line("no-json\n") == {}
    assert mod.token_counts_for("two words", "one", {}) == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
        "source": "whitespace_estimate",
    }
    assert hashlib.sha256(b"fixture").hexdigest() == mod.sha256_bytes(b"fixture")
    assert mod.fake_evidence_rejection_criteria({})[0].startswith("reject no model loaded")
    assert mod._mapping_list("not-list") == []
    assert mod._float_matches(None, 1.0) is False

    selected_model = {
        "hf_id": MODEL_ID,
        "model_path": str(model_path),
        "path_exists": True,
    }
    no_gpu_smoke = mod.maybe_run_replay(
        selected_python="/venv/python",
        selected_model=selected_model,
        substrate_probe={"cuda_available": False, "gpu_count": 0},
        contract_ready=True,
        command_runner=_runner(),
        random_seed=mod.DEFAULT_RANDOM_SEED,
    )
    assert no_gpu_smoke["model_load_evidence"]["load_attempted"] is False
    runtime_error_smoke = mod.maybe_run_replay(
        selected_python="/venv/python",
        selected_model=selected_model,
        substrate_probe={"cuda_available": True, "gpu_count": 1},
        contract_ready=True,
        command_runner=_runner({"ok": False, "error": "fixture runtime failed"}),
        random_seed=mod.DEFAULT_RANDOM_SEED,
    )
    assert runtime_error_smoke["runtime_blocker"] == "fixture runtime failed"
    empty_output_smoke = mod.maybe_run_replay(
        selected_python="/venv/python",
        selected_model=selected_model,
        substrate_probe={"cuda_available": True, "gpu_count": 1},
        contract_ready=True,
        command_runner=_runner(
            {
                "ok": True,
                "runtime": "llama_cpp",
                "load_wall_time_s": 1.0,
                "total_worker_wall_time_s": 1.5,
                "calls": [{"prompt": mod.DEFAULT_PROMPTS[0], "output_text": ""}],
            }
        ),
        random_seed=mod.DEFAULT_RANDOM_SEED,
    )
    assert empty_output_smoke["runtime_blocker"] == "smoke worker produced no replay transcripts"
    failed_load = mod.load_evidence_from_result(
        selected_model,
        ["cmd"],
        {"returncode": 1, "stdout": "{}", "stderr": "boom"},
        {"runtime": "llama_cpp"},
    )
    assert failed_load["runtime_error"] == "boom"
    fallback_seed_rows = mod.transcript_hash_rows(
        selected_model,
        [{"prompt": "p", "output_text": "o", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}],
        17,
    )
    assert fallback_seed_rows[0]["random_seed"] == 17

    bad_live = saved | {
        "live_call_count": 2,
        "model_specs": [],
        "model_load_evidence": {
            "load_attempted": False,
            "path_exists": False,
            "load_wall_time_s": None,
            "generation_wall_time_s": 0.0,
            "stdout_summary": "{}",
        },
        "transcript_hashes": [
            {"transcript_sha256": "dup", "response_hash": ""},
            {"transcript_sha256": "dup", "response_hash": "r"},
        ],
        "token_counts": {"prompt_tokens": 1, "completion_tokens": 1},
        "random_seed": None,
        "reproducibility_checksum": "",
        "controlled_subprocess_return_codes": [{"returncode": 1}],
    }
    assert mod.fake_evidence_violations(bad_live) == [
        "no model loaded",
        "missing transcript hashes",
        "missing seed/checksum",
        "impossible token throughput",
        "reused stale transcript hash",
        "selected model/local path mismatch",
        "wall-clock claims not supported by command output",
        "uncontrolled subprocess outcomes",
    ]
    assert mod.determine_blocked_reason(
        contract_ready=False,
        usable_ids=[MODEL_ID],
        substrate_probe={"cuda_available": True, "gpu_count": 1},
        smoke_blocker="",
        artifact=saved,
    ).startswith("duration-corrected")
    assert mod.determine_blocked_reason(
        contract_ready=True,
        usable_ids=[MODEL_ID],
        substrate_probe={"cuda_available": False, "gpu_count": 0},
        smoke_blocker="",
        artifact=saved,
    ).startswith("CUDA/GPU")
    assert mod.determine_blocked_reason(
        contract_ready=True,
        usable_ids=[MODEL_ID],
        substrate_probe={"cuda_available": True, "gpu_count": 1},
        smoke_blocker="runtime blocker",
        artifact=saved,
    ) == "runtime blocker"
    assert mod.determine_blocked_reason(
        contract_ready=True,
        usable_ids=[MODEL_ID],
        substrate_probe={"cuda_available": True, "gpu_count": 1},
        smoke_blocker="",
        artifact=saved | {"measured_work_policy_passed": False},
    ).startswith("measured-work")
    assert mod.determine_blocked_reason(
        contract_ready=True,
        usable_ids=[MODEL_ID],
        substrate_probe={"cuda_available": True, "gpu_count": 1},
        smoke_blocker="",
        artifact=saved | {
            "measured_work_policy_passed": True,
            "token_scaled_duration_policy_passed": False,
        },
    ).startswith("token-scaled")
    assert mod.determine_blocked_reason(
        contract_ready=True,
        usable_ids=[MODEL_ID],
        substrate_probe={"cuda_available": True, "gpu_count": 1},
        smoke_blocker="",
        artifact=saved | {
            "measured_work_policy_passed": True,
            "token_scaled_duration_policy_passed": True,
            "repeated_call_policy_passed": True,
            "fake_evidence_rejection_passed": False,
            "fake_evidence_rejection_violations": ["bad"],
        },
    ) == "fake-evidence rejection failed: bad"
    assert mod.honest_verdict({"preflight_passed": False, "blocked_reason": "contract failed"}).startswith(
        "blocked_contract_precondition:"
    )
    assert mod.honest_verdict({"preflight_passed": False, "blocked_reason": "CUDA/GPU substrate"}).startswith(
        "blocked_gpu_substrate:"
    )
    assert mod.honest_verdict({"preflight_passed": False, "blocked_reason": "fake-evidence failed"}).startswith(
        "blocked_fake_evidence_rejection:"
    )
    assert mod.honest_verdict({"preflight_passed": False, "blocked_reason": "token-scaled failed"}).startswith(
        "blocked_token_scaled_duration_policy:"
    )
    assert mod.honest_verdict({"preflight_passed": False, "blocked_reason": "other"}).startswith(
        "blocked_smoke_runtime:"
    )

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="headline"):
        mod.validate_artifact(saved | {"headline_claim_allowed": True})
    with pytest.raises(ValueError, match="live call count"):
        mod.validate_artifact(saved | {"preflight_passed": True, "live_call_count": 0})
    with pytest.raises(ValueError, match="transcript_hashes"):
        mod.validate_artifact(saved | {"preflight_passed": True, "transcript_hashes": []})
    with pytest.raises(ValueError, match="terminal success prefix"):
        mod.validate_artifact(saved | {"honest_verdict": "blocked_wrong:"})
    with pytest.raises(ValueError, match="blocked_"):
        mod.validate_artifact(saved | {"preflight_passed": False, "honest_verdict": "complete: wrong"})
