"""Tests for Exp 3179 local SOTA receipt smoke v3.

Spec refs: REQ-VERIFY-3179, SCENARIO-VERIFY-3179.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import local_sota_receipt_smoke_v3 as mod


QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31_ID = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "source\n") -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _model_path(cache_root: Path, hf_id: str, filename: str) -> Path:
    owner, name = hf_id.split("/", 1)
    return cache_root / f"models--{owner}--{name}" / "snapshots" / "abc123" / filename


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
        "openspec/capabilities/verification/spec.md",
        "REQ-VERIFY-3179\nSCENARIO-VERIFY-3179\n"
        "results/experiment_3179_local_sota_receipt_smoke_v3.json\n"
        "full_local_sota_receipt\ncpu_fallback_receipt_only\n",
    )
    _write_json(
        root,
        mod.EXP3178_REL_PATH,
        {
            "artifact": "experiment_3178_receipt_backed_authenticity_contract_v3",
            "receipt_backed_authenticity_contract_v3_ready": True,
            "required_receipt_fields": list(mod.REQUIRED_RECEIPT_FIELDS),
            "clean_rerun_unlock_requirements": [
                "exp3179.substrate_classification=full_local_sota_receipt",
                "throughput_plausibility.passed=true",
            ],
            "substrate_classification_policy": {
                "classes": {name: {} for name in mod.SUBSTRATE_CLASSES}
            },
            "prior_receipts": [{"transcript_sha256": "stale-prior-hash"}],
            "honest_verdict": "complete: receipt_backed_authenticity_contract_v3_ready=true",
        },
    )


def _cached_pair(ids: list[str]):
    def provider() -> list[dict[str, str]] | None:
        if len(ids) < 2:
            return None
        return [{"hf_id": hf_id, "model_path": f"/cache/{hf_id}.gguf"} for hf_id in ids[:2]]

    return provider


def _worker_payload() -> dict[str, Any]:
    return {
        "ok": True,
        "runtime": "llama_cpp",
        "load_wall_time_s": 2.5,
        "total_worker_wall_time_s": 3.75,
        "calls": [
            {
                "prompt": mod.DEFAULT_PROMPTS[0],
                "seed": mod.DEFAULT_RANDOM_SEED,
                "output_text": "READY",
                "generation_wall_time_s": 0.4,
                "usage": {"prompt_tokens": 6, "completion_tokens": 1, "total_tokens": 7},
            },
            {
                "prompt": mod.DEFAULT_PROMPTS[1],
                "seed": mod.DEFAULT_RANDOM_SEED + 1,
                "output_text": "VERIFIED",
                "generation_wall_time_s": 0.5,
                "usage": {"prompt_tokens": 6, "completion_tokens": 1, "total_tokens": 7},
            },
        ],
    }


def _runner(
    *,
    loader_ok: bool = True,
    cuda_available: bool = True,
    worker_payload: dict[str, Any] | None = None,
):
    def run(command: list[str], *, timeout_s: int = 10, env: dict[str, str] | None = None):
        del timeout_s, env
        joined = "\n".join(command)
        if "exp3179_loader_probe" in joined:
            return {
                "command": command,
                "returncode": 0 if loader_ok else 1,
                "stdout": json.dumps(
                    {
                        "ok": loader_ok,
                        "loader_name": "llama_cpp.Llama" if loader_ok else None,
                        "version": "0.3.fixture" if loader_ok else None,
                        "error": "" if loader_ok else "ModuleNotFoundError: llama_cpp",
                    },
                    sort_keys=True,
                )
                + "\n",
                "stderr": "" if loader_ok else "no module named llama_cpp\n",
            }
        if "exp3179_torch_cuda_probe" in joined:
            return {
                "command": command,
                "returncode": 0,
                "stdout": json.dumps(
                    {
                        "torch_present": True,
                        "torch_import_ok": True,
                        "torch_version": "2.11.0+cu128",
                        "cuda_available": cuda_available,
                        "device_count": 1 if cuda_available else 0,
                        "cuda_version": "12.8" if cuda_available else None,
                    },
                    sort_keys=True,
                )
                + "\n",
                "stderr": "",
            }
        if command and command[0] == "nvidia-smi":
            return {
                "command": command,
                "returncode": 0 if cuda_available else 127,
                "stdout": "0, NVIDIA GeForce RTX 3090, 24576\n" if cuda_available else "",
                "stderr": "" if cuda_available else "nvidia-smi unavailable\n",
            }
        if "--exp3179-smoke-worker" in command:
            payload = worker_payload or _worker_payload()
            return {
                "command": command,
                "returncode": 0 if payload.get("ok") else 1,
                "stdout": json.dumps(payload, sort_keys=True) + "\n",
                "stderr": "llama.cpp fixture stderr tail\n",
            }
        raise AssertionError(f"unexpected command: {command}")

    return run


def test_req_verify_3179_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3179: OpenSpec declares the local receipt smoke before code."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3179" in spec
    assert "SCENARIO-VERIFY-3179" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "full_local_sota_receipt" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3179_full_cuda_receipt_unlocks_clean_rerun(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3179: CUDA-backed mandated receipts unlock Exp 3181 attempt."""

    cache_root = tmp_path / "hf-cache"
    qwen_path = _model_path(cache_root, QWEN_ID, "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    qwen_path.parent.mkdir(parents=True)
    qwen_path.write_bytes(b"qwen gguf fixture")
    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python="/venv/python",
        command_runner=_runner(cuda_available=True),
        cached_pair_provider=_cached_pair([QWEN_ID, GEMMA26_ID]),
        started_s=10.0,
        now_s=15.0,
        tests_run=["SCENARIO-VERIFY-3179 focused"],
    )

    assert artifact["local_sota_receipt_smoke_v3_ready"] is True
    assert artifact["preflight_passed"] is True
    assert artifact["live_call_count"] == 2
    assert artifact["selected_model_ids"] == [QWEN_ID]
    assert artifact["substrate_classification"] == "full_local_sota_receipt"
    assert artifact["cpu_fallback_used"] is False
    assert artifact["throughput_plausibility_passed"] is True
    assert artifact["headline_claim_allowed"] is False
    assert artifact["clean_rerun_allowed"] is True
    assert artifact["honest_verdict"].startswith("complete:")

    assert len(artifact["mandated_model_inventory"]) == 3
    assert artifact["cached_sota_pair_probe"]["returned_pair"] is True
    assert artifact["loader_probe"]["available"] is True
    assert artifact["cuda_probe"]["cuda_available"] is True
    assert artifact["inference_substrate"]["n_gpu_layers"] == -1
    assert artifact["inference_substrate"]["live_model_calls"] == 2
    assert artifact["proof_receipts"][0]["selected_model_id"] == QWEN_ID
    assert artifact["proof_receipts"][0]["model_path"] == str(qwen_path)
    assert artifact["proof_receipts"][0]["subprocess_return_code"] == 0
    assert artifact["proof_receipts"][0]["stderr_tail"] == "llama.cpp fixture stderr tail"
    assert len(set(artifact["prompt_hashes"])) == 2
    assert len(set(artifact["transcript_hashes"])) == 2
    assert artifact["token_counts"]["total_tokens"] == 14


def test_scenario_verify_3179_cpu_fallback_is_receipt_only(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3179: CPU smoke can prove wiring but cannot unlock reruns."""

    cache_root = tmp_path / "hf-cache"
    gemma31_path = _model_path(cache_root, GEMMA31_ID, "gemma-4-31B-it-UD-Q4_K_M.gguf")
    gemma26_path = _model_path(cache_root, GEMMA26_ID, "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")
    gemma31_path.parent.mkdir(parents=True)
    gemma26_path.parent.mkdir(parents=True)
    gemma31_path.write_bytes(b"gemma31 gguf fixture")
    gemma26_path.write_bytes(b"gemma26 gguf fixture")
    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python="/venv/python",
        command_runner=_runner(cuda_available=False),
        cached_pair_provider=_cached_pair([GEMMA26_ID, GEMMA31_ID]),
    )

    assert artifact["preflight_passed"] is True
    assert artifact["live_call_count"] == 2
    assert artifact["selected_model_ids"] == [GEMMA31_ID]
    assert artifact["substrate_classification"] == "cpu_fallback_receipt_only"
    assert artifact["cpu_fallback_used"] is True
    assert artifact["headline_claim_allowed"] is False
    assert artifact["clean_rerun_allowed"] is False
    assert artifact["inference_substrate"]["n_gpu_layers"] == 0
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_3179_blocks_without_mandated_cache(tmp_path: Path) -> None:
    """REQ-VERIFY-3179: missing mandated GGUFs produce a no-call blocked artifact."""

    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        cache_root=tmp_path / "empty-cache",
        selected_python="/venv/python",
        command_runner=_runner(cuda_available=True),
        cached_pair_provider=_cached_pair([]),
    )

    assert artifact["local_sota_receipt_smoke_v3_ready"] is True
    assert artifact["preflight_passed"] is False
    assert artifact["live_call_count"] == 0
    assert artifact["selected_model_ids"] == []
    assert artifact["substrate_classification"] == "model_cache_missing"
    assert artifact["proof_receipts"] == []
    assert artifact["throughput_plausibility_passed"] is False
    assert artifact["headline_claim_allowed"] is False
    assert artifact["clean_rerun_allowed"] is False
    assert artifact["honest_verdict"].startswith("blocked_model_cache_missing:")
    assert all(row["cache_status"] == "missing" for row in artifact["mandated_model_inventory"])


def test_req_verify_3179_blocks_when_loader_missing(tmp_path: Path) -> None:
    """REQ-VERIFY-3179: a cache hit without llama_cpp is a loader block."""

    cache_root = tmp_path / "hf-cache"
    qwen_path = _model_path(cache_root, QWEN_ID, "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    qwen_path.parent.mkdir(parents=True)
    qwen_path.write_bytes(b"qwen gguf fixture")
    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python="/venv/python",
        command_runner=_runner(loader_ok=False, cuda_available=True),
        cached_pair_provider=_cached_pair([QWEN_ID, GEMMA26_ID]),
    )

    assert artifact["preflight_passed"] is False
    assert artifact["live_call_count"] == 0
    assert artifact["selected_model_ids"] == [QWEN_ID]
    assert artifact["substrate_classification"] == "loader_missing"
    assert artifact["loader_probe"]["available"] is False
    assert artifact["proof_receipts"] == []
    assert artifact["honest_verdict"].startswith("blocked_loader_missing:")


def test_req_verify_3179_writer_stale_hash_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3179: writer persists JSON and validators reject overclaims."""

    cache_root = tmp_path / "hf-cache"
    qwen_path = _model_path(cache_root, QWEN_ID, "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    qwen_path.parent.mkdir(parents=True)
    qwen_path.write_bytes(b"qwen gguf fixture")
    _write_common_sources(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python="/venv/python",
        command_runner=_runner(cuda_available=True),
        cached_pair_provider=_cached_pair([QWEN_ID, GEMMA26_ID]),
        started_s=1.0,
        now_s=4.0,
        tests_run=["focused"],
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["duration_s"] == pytest.approx(3.0)
    assert saved["tests_run"] == ["focused"]
    assert mod.duration(5.0, 3.0) == 0.0
    assert mod.hash_text("x") == mod.hash_text("x")
    assert mod.first_json_line("bad\n{\"ok\": true}\n") == {"ok": True}
    assert mod.first_json_line("bad\n") == {}
    assert mod.truncate_tail("abcdef", limit=3) == "def"
    assert mod.safe_float("x") is None
    assert mod.safe_int("x") is None
    assert mod.mapping_list("x") == []
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad json}\n", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    assert mod.probe_cached_sota_pair(lambda: (_ for _ in ()).throw(RuntimeError("boom"))) == {
        "called": True,
        "returned_pair": False,
        "model_ids": [],
        "error": "RuntimeError: boom",
    }
    relative = tmp_path / "relative.gguf"
    relative.write_bytes(b"relative")
    assert mod.path_evidence(tmp_path, "relative.gguf")["exists"] is True
    assert mod.path_evidence(tmp_path, "missing.gguf")["exists"] is False
    assert mod.token_counts_for("two words", "one", {}) == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
        "source": "whitespace_estimate",
    }
    assert mod.receipt_throughput_plausible(
        {"token_counts": {"completion_tokens": 0}, "wall_clock_s": None}
    ) is True

    stale_hash = mod.transcript_hash(
        QWEN_ID,
        saved["proof_receipts"][0]["prompt_hash"],
        saved["proof_receipts"][0]["response_hash"],
        saved["proof_receipts"][0]["random_seed"],
    )
    assert stale_hash == saved["proof_receipts"][0]["transcript_hash"]
    assert mod.receipts_are_fresh(
        [{"transcript_hash": stale_hash}, {"transcript_hash": stale_hash}],
        {"other"},
    ) is False
    assert mod.receipts_are_fresh([{"transcript_hash": stale_hash}], {stale_hash}) is False
    stale_contract = json.loads((tmp_path / mod.EXP3178_REL_PATH).read_text(encoding="utf-8"))
    stale_contract["prior_receipts"] = [{"transcript_sha256": stale_hash}]
    _write_json(tmp_path, mod.EXP3178_REL_PATH, stale_contract)
    stale_artifact = mod.build_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python="/venv/python",
        command_runner=_runner(cuda_available=True),
        cached_pair_provider=_cached_pair([QWEN_ID, GEMMA26_ID]),
    )
    assert stale_artifact["preflight_passed"] is False
    assert stale_artifact["stale_transcript_rejection_passed"] is False
    assert stale_artifact["blocked_reason"] == "reused stale transcript hash"
    stale_contract["prior_receipts"] = []
    _write_json(tmp_path, mod.EXP3178_REL_PATH, stale_contract)

    bad_payload = _worker_payload()
    bad_payload["calls"][0]["generation_wall_time_s"] = 0.0
    impossible = mod.build_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python="/venv/python",
        command_runner=_runner(cuda_available=True, worker_payload=bad_payload),
        cached_pair_provider=_cached_pair([QWEN_ID, GEMMA26_ID]),
    )
    assert impossible["preflight_passed"] is False
    assert impossible["throughput_plausibility_passed"] is False
    assert impossible["honest_verdict"].startswith("blocked_throughput_plausibility:")

    runtime_failed = mod.build_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python="/venv/python",
        command_runner=_runner(
            cuda_available=True,
            worker_payload={"ok": False, "error": "fixture worker failed", "calls": []},
        ),
        cached_pair_provider=_cached_pair([QWEN_ID, GEMMA26_ID]),
    )
    assert runtime_failed["blocked_reason"] == "fixture worker failed"
    assert runtime_failed["honest_verdict"].startswith("blocked_cuda_available_unhealthy:")

    empty_payload = _worker_payload()
    empty_payload["calls"] = [{"prompt": mod.DEFAULT_PROMPTS[0], "output_text": ""}]
    empty_artifact = mod.build_artifact(
        tmp_path,
        cache_root=cache_root,
        selected_python="/venv/python",
        command_runner=_runner(cuda_available=True, worker_payload=empty_payload),
        cached_pair_provider=_cached_pair([QWEN_ID, GEMMA26_ID]),
    )
    assert empty_artifact["blocked_reason"] == "smoke worker produced no proof receipts"

    assert mod.blocked_reason_for(
        v3_contract_ready=False,
        selected_model={"hf_id": QWEN_ID},
        loader_probe={"available": True},
        smoke={"runtime_blocker": ""},
        live_call_count=2,
        receipts_fresh=True,
        throughput_passed=True,
    ).startswith("Exp 3178")
    assert mod.blocked_reason_for(
        v3_contract_ready=True,
        selected_model={"hf_id": QWEN_ID},
        loader_probe={"available": True},
        smoke={"runtime_blocker": ""},
        live_call_count=1,
        receipts_fresh=True,
        throughput_passed=True,
    ).startswith("receipt smoke produced 1")
    assert mod.blocked_reason_for(
        v3_contract_ready=True,
        selected_model={"hf_id": QWEN_ID},
        loader_probe={"available": True},
        smoke={"runtime_blocker": ""},
        live_call_count=2,
        receipts_fresh=False,
        throughput_passed=True,
    ) == "reused stale transcript hash"
    assert mod.honest_verdict(
        {
            "preflight_passed": False,
            "blocked_reason": "CUDA missing",
            "substrate_classification": "cuda_unavailable",
            "live_call_count": 0,
        }
    ).startswith("blocked_cuda_unavailable:")
    assert mod.honest_verdict(
        {
            "preflight_passed": False,
            "blocked_reason": "other",
            "substrate_classification": "cpu_fallback_receipt_only",
            "live_call_count": 0,
        }
    ).startswith("blocked_receipt_precondition:")

    broken = dict(saved)
    del broken["proof_receipts"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["substrate_classification"] = "wrong"
    with pytest.raises(ValueError, match="substrate_classification"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["headline_claim_allowed"] = True
    with pytest.raises(ValueError, match="headline"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["clean_rerun_allowed"] = True
    broken["substrate_classification"] = "cpu_fallback_receipt_only"
    with pytest.raises(ValueError, match="clean rerun"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["preflight_passed"] = True
    broken["live_call_count"] = 0
    with pytest.raises(ValueError, match="live_call_count"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["preflight_passed"] = False
    broken["honest_verdict"] = "complete: wrong"
    with pytest.raises(ValueError, match="blocked_"):
        mod.validate_artifact(broken)

    broken = dict(saved)
    broken["honest_verdict"] = "maybe"
    with pytest.raises(ValueError, match="terminal"):
        mod.validate_artifact(broken)
