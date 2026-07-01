"""Tests for Exp 5097 clean SOTA endpoint/logprob cache provenance.

Spec refs: REQ-INFER-SOTA-027,
SCENARIO-INFER-SOTA-027-SUCCESS,
SCENARIO-INFER-SOTA-027-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5097_clean_sota_endpoint_logprob_cache as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "llm-ebm-inference" / "spec.md"
QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _write_models(tmp_path: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    sizes = {QWEN: 30, GEMMA31: 20, GEMMA26: 10}
    for hf_id, size in sizes.items():
        path = tmp_path / "models" / hf_id.split("/", 1)[1] / f"{hf_id.split('/', 1)[1]}-Q4_K_M.gguf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x" * size, encoding="utf-8")
        paths[hf_id] = path.as_posix()
    return paths


def _resolver(paths: dict[str, str]) -> mod.ModelResolver:
    return lambda hf_id, preferred_quant: paths.get(hf_id)


def _cached_pair(paths: dict[str, str]) -> mod.CachedPairFn:
    def fake(*, gpu_indices: tuple[int, int], preferred_quant: str) -> list[dict[str, Any]]:
        del preferred_quant
        return [
            {"name": "Qwen3.6-35B-A3B", "hf_id": QWEN, "gpu": gpu_indices[0], "model_path": paths[QWEN]},
            {"name": "Gemma4-26B-A4B-it", "hf_id": GEMMA26, "gpu": gpu_indices[1], "model_path": paths[GEMMA26]},
        ]

    return fake


def _preconditions() -> dict[str, Any]:
    return {
        "cuda_gpu_visibility": {
            "cuda_available": True,
            "gpu_count": 2,
            "gpus": [
                {"index": 0, "name": "RTX 3090", "free_vram_mb": 22000},
                {"index": 1, "name": "RTX 3090", "free_vram_mb": 21900},
            ],
        },
        "llama_cpp_python": {"available": True, "detail": "llama_cpp import ok"},
        "free_vram": {"available": True, "total_free_vram_mb": 43900},
    }


def _server_unavailable(_env: dict[str, str]) -> dict[str, Any]:
    return {
        "available": False,
        "selected_path": None,
        "candidates": [
            {
                "source": "test",
                "path": "/missing/llama-server",
                "exists": False,
                "is_file": False,
                "executable": False,
            }
        ],
        "missing_diagnostic": "llama-server binary not found or not executable",
    }


def _server_available(path: Path) -> dict[str, Any]:
    return {
        "available": True,
        "selected_path": path.as_posix(),
        "candidates": [
            {
                "source": "test",
                "path": path.as_posix(),
                "exists": True,
                "is_file": True,
                "executable": True,
            }
        ],
        "missing_diagnostic": None,
    }


def _free_port(port: int = 45697) -> dict[str, Any]:
    return {
        "available": True,
        "host": "127.0.0.1",
        "port": port,
        "endpoint_url": f"http://127.0.0.1:{port}",
        "error": None,
    }


def _blocked_probe(endpoints: list[str], timeout_s: float) -> dict[str, Any]:
    return {
        "candidate_endpoints": list(endpoints),
        "selected_endpoint": None,
        "completion_ready": False,
        "top_logprob_ready": False,
        "confidence_ready": False,
        "telemetry_signal": None,
        "duration_s": timeout_s,
        "probes": [
            {
                "endpoint": endpoints[0],
                "completion_probe": {"ready": False, "detail": "connection refused"},
            }
        ],
    }


def _ready_sample(endpoint: str, timeout_s: float) -> dict[str, Any]:
    del timeout_s
    return {
        "ready": True,
        "route": endpoint.rstrip("/") + "/completion",
        "status": 200,
        "completion_text": "exp5097 endpoint live",
        "logprob_ready": True,
        "top_logprob_ready": True,
        "confidence_ready": False,
        "telemetry_signal": "top_logprobs",
        "evidence": {
            "token_logprob_count": 2,
            "top_logprob_row_count": 2,
            "token_logprobs": [-0.11, -0.22],
            "top_logprobs": [{" exp": -0.11, " run": -1.3}, {" live": -0.22, " blocked": -2.0}],
            "raw_response_keys": ["content", "completion_probabilities"],
        },
        "error": None,
    }


def _cache_sample(row_index: int, endpoint: str, selected_model: dict[str, Any]) -> dict[str, Any]:
    del selected_model
    return {
        "ready": True,
        "route": endpoint.rstrip("/") + "/completion",
        "status": 200,
        "completion_text": f"cache row {row_index}",
        "logprob_ready": True,
        "top_logprob_ready": True,
        "confidence_ready": False,
        "telemetry_signal": "top_logprobs",
        "evidence": {
            "token_logprob_count": 2,
            "top_logprob_row_count": 1,
            "token_logprobs": [-0.1 - row_index / 1000, -0.2 - row_index / 1000],
            "top_logprobs": [{f"row-{row_index}": -0.1, "alt": -1.0}],
            "raw_response_keys": ["content", "completion_probabilities"],
        },
        "error": None,
    }


def _clean_adversarial(_path: Path) -> dict[str, Any]:
    return {"flags": [], "summary": {"critical_count": 0}}


def test_req_infer_sota_027_spec_declares_exp5097_contract() -> None:
    """REQ-INFER-SOTA-027: OpenSpec anchors fields, paths, scenarios, and principles."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-INFER-SOTA-027",
        "SCENARIO-INFER-SOTA-027-SUCCESS",
        "SCENARIO-INFER-SOTA-027-BLOCKED",
        "python/carnot/experiment_5097_clean_sota_endpoint_logprob_cache.py",
        "results/experiment_5097_clean_sota_endpoint_logprob_cache_v468.json",
        "results/experiment_5097_clean_sota_endpoint_logprob_cache_v468.jsonl",
        "success_clean_sota_endpoint_logprob_cache_ready",
        "blocked_clean_sota_endpoint_logprob_cache_no_live_logprobs",
        QWEN,
        GEMMA31,
        GEMMA26,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_infer_sota_027_blocked_records_preconditions_without_cache_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-027-BLOCKED: no live logprobs means no cache promotion."""

    paths = _write_models(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        cache_path=tmp_path / mod.CACHE_RELATIVE_PATH,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        precondition_probe=lambda root, env: _preconditions(),
        endpoint_probe=_blocked_probe,
        endpoint_sample=lambda endpoint, timeout_s: pytest.fail("sample must not run"),
        cache_sample=lambda row_index, endpoint, selected_model: pytest.fail("cache must not run"),
        server_finder=_server_unavailable,
        free_port=lambda host: _free_port(),
        adversarial_verify=_clean_adversarial,
        now=iter([10.0, 11.25]).__next__,
        duration_floor_s=0.0,
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_clean_sota_endpoint_logprob_cache_no_live_logprobs"
    assert artifact["inference_substrate"] == "precondition_check_only"
    assert artifact["completion_endpoint_ready"] is False
    assert artifact["logprob_endpoint_ready"] is False
    assert artifact["top_logprob_or_confidence_ready"] is False
    assert artifact["logprob_endpoint_clean"] is False
    assert artifact["live_llm_invoked"] is False
    assert artifact["cache_rows_written"] == 0
    assert not (tmp_path / mod.CACHE_RELATIVE_PATH).exists()
    assert artifact["blocker_root_cause"]["kind"] == "llama_server_binary_unavailable"
    assert artifact["preconditions_checked"]["free_port"]["available"] is True
    assert artifact["preconditions_checked"]["cache_path"]["path"].endswith(".jsonl")
    assert artifact["preconditions_checked"]["free_vram"]["total_free_vram_mb"] == 43900
    assert set(artifact["preconditions_checked"]["resolved_local_gguf_paths"]) == {
        QWEN,
        GEMMA31,
        GEMMA26,
    }
    assert [row["hf_id"] for row in artifact["model_specs"]["mandatory_models"]] == [
        QWEN,
        GEMMA31,
        GEMMA26,
    ]
    assert [row["hf_id"] for row in artifact["usable_sota_models"]] == [QWEN, GEMMA31, GEMMA26]
    assert artifact["server_command"] is None
    assert artifact["adversarial_verify_passed"] is True
    assert artifact["flagged_adversarial"] is False
    assert artifact["claimed_capability_scope"] == "runtime_provenance_only"
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_infer_sota_027_success_records_server_lifetime_and_ten_cache_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-027-SUCCESS: live top-logprobs write a clean smoke cache."""

    paths = _write_models(tmp_path)
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    start_calls: list[dict[str, Any]] = []
    cleanup_calls: list[Any] = []
    probe_calls: list[list[str]] = []

    class FakeProcess:
        pid = 5097

        def poll(self) -> None:
            return None

    def probe(endpoints: list[str], timeout_s: float) -> dict[str, Any]:
        probe_calls.append(list(endpoints))
        ready = endpoints == ["http://127.0.0.1:45697"]
        return {
            "candidate_endpoints": list(endpoints),
            "selected_endpoint": endpoints[0] if ready else None,
            "completion_ready": ready,
            "top_logprob_ready": ready,
            "confidence_ready": False,
            "telemetry_signal": "top_logprobs" if ready else None,
            "duration_s": timeout_s,
            "probes": [],
        }

    def start(command: list[str], env: dict[str, str], log_path: Path) -> FakeProcess:
        start_calls.append({"command": command, "env": env, "log_path": log_path})
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("llama server ready\n", encoding="utf-8")
        return FakeProcess()

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        cache_path=tmp_path / mod.CACHE_RELATIVE_PATH,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        precondition_probe=lambda root, env: _preconditions(),
        endpoint_probe=probe,
        endpoint_sample=_ready_sample,
        cache_sample=_cache_sample,
        server_finder=lambda env: _server_available(server),
        free_port=lambda host: _free_port(45697),
        server_start=start,
        server_cleanup=lambda process: cleanup_calls.append(process) or {"started_by_preflight": True, "terminated": True, "returncode": 0},
        adversarial_verify=_clean_adversarial,
        now=iter([100.0, 164.0]).__next__,
        duration_floor_s=0.0,
        write=True,
    )
    rows = mod.read_jsonl_rows(tmp_path / mod.CACHE_RELATIVE_PATH)

    assert artifact["honest_verdict"] == "success_clean_sota_endpoint_logprob_cache_ready"
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["completion_endpoint_ready"] is True
    assert artifact["logprob_endpoint_ready"] is True
    assert artifact["top_logprob_or_confidence_ready"] is True
    assert artifact["logprob_endpoint_clean"] is True
    assert artifact["live_llm_invoked"] is True
    assert artifact["endpoint_url"] == "http://127.0.0.1:45697"
    assert artifact["endpoint_lifetime_s"] == pytest.approx(64.0)
    assert artifact["server_pid"] == 5097
    assert artifact["server_command"] == start_calls[0]["command"]
    assert artifact["server_command"][0] == server.as_posix()
    assert artifact["server_command"][1:3] == ["-m", paths[GEMMA26]]
    assert artifact["server_logs"]["tail"] == "llama server ready\n"
    assert artifact["cleanup_behavior"]["terminated"] is True
    assert cleanup_calls and cleanup_calls[0].pid == 5097
    assert probe_calls[0] == ["http://127.0.0.1:8080"]
    assert probe_calls[1] == ["http://127.0.0.1:45697"]
    assert artifact["sample_completion"]["text"] == "exp5097 endpoint live"
    assert artifact["sample_logprob_evidence"]["token_logprob_count"] == 2
    assert artifact["sample_logprob_evidence"]["top_logprob_row_count"] == 2
    assert artifact["cache_rows_written"] == 10
    assert len(rows) == 10
    assert all(mod.validate_cache_row(row) == [] for row in rows)
    assert {row["model_hf_id"] for row in rows} == {GEMMA26}
    assert artifact["adversarial_verify_passed"] is True
    assert artifact["flagged_adversarial"] is False
    assert artifact["blocker_root_cause"] is None
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact


def test_req_infer_sota_027_committed_artifact_is_schema_valid() -> None:
    """REQ-INFER-SOTA-027: the checked-in deliverable satisfies the clean schema."""

    artifact_path = REPO / mod.RESULT_RELATIVE_PATH
    assert artifact_path.exists()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert mod.artifact_schema_errors(artifact) == []
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert all(field in artifact["field_principles"] for field in mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["claimed_capability_scope"] == "runtime_provenance_only"
    assert "uprm" not in artifact["honest_verdict"].lower()
    assert "hallucination" not in artifact["honest_verdict"].lower()
    if artifact["cache_rows_written"]:
        rows = mod.read_jsonl_rows(Path(artifact["cache_path"]))
        assert len(rows) == artifact["cache_rows_written"]
        assert all(mod.validate_cache_row(row) == [] for row in rows)
    else:
        assert artifact["cache_rows_written"] == 0
        assert artifact["blocker_root_cause"]
