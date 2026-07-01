"""Tests for Exp 5124 clean local SOTA runtime provenance.

Spec refs: REQ-INFER-SOTA-029,
SCENARIO-INFER-SOTA-029-CLEAN,
SCENARIO-INFER-SOTA-029-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5124_clean_sota_runtime_provenance as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "llm-ebm-inference" / "spec.md"
QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _write_models(tmp_path: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    sizes = {QWEN: 30, GEMMA31: 20, GEMMA26: 10}
    for hf_id, size in sizes.items():
        model_name = hf_id.split("/", 1)[1]
        path = tmp_path / "models" / model_name / f"{model_name}-Q4_K_M.gguf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x" * size, encoding="utf-8")
        paths[hf_id] = path.as_posix()
    return paths


def _resolver(paths: dict[str, str]) -> mod.ModelResolver:
    return lambda hf_id, preferred_quant: paths.get(hf_id)


def _cached_pair(paths: dict[str, str]) -> mod.CachedPairFn:
    def fake(*, gpu_indices: tuple[int, int], preferred_quant: str) -> list[dict[str, Any]]:
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": QWEN,
                "gpu": gpu_indices[0],
                "model_path": paths[QWEN],
                "preferred_quant": preferred_quant,
            },
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": GEMMA26,
                "gpu": gpu_indices[1],
                "model_path": paths[GEMMA26],
                "preferred_quant": preferred_quant,
            },
        ]

    return fake


def _preconditions() -> dict[str, Any]:
    return {
        "cuda_status": {
            "cuda_available": True,
            "gpu_count": 2,
            "gpus": [
                {"index": 0, "name": "RTX 3090", "free_vram_mb": 22000},
                {"index": 1, "name": "RTX 3090", "free_vram_mb": 21900},
            ],
        },
        "llama_cpp_python": {"available": True, "detail": "llama_cpp import ok"},
        "disk_ram": {"disk_free_gib": 100.0, "ram_available_gib": 64.0},
    }


def _server_unavailable(_env: dict[str, str]) -> dict[str, Any]:
    return {
        "available": False,
        "selected_path": None,
        "candidates": [],
        "missing_diagnostic": "llama-server binary not found or not executable",
    }


def _free_port(port: int = 45124) -> dict[str, Any]:
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
                "telemetry_probe": {"ready": False, "detail": "completion probe failed"},
            }
        ],
    }


def _ready_probe(endpoints: list[str], timeout_s: float) -> dict[str, Any]:
    del timeout_s
    return {
        "candidate_endpoints": list(endpoints),
        "selected_endpoint": endpoints[0],
        "completion_ready": True,
        "top_logprob_ready": True,
        "confidence_ready": False,
        "telemetry_signal": "top_logprobs",
        "duration_s": 0.25,
        "probes": [],
    }


def _ready_sample(endpoint: str, timeout_s: float) -> dict[str, Any]:
    del timeout_s
    return {
        "ready": True,
        "route": endpoint.rstrip("/") + "/completion",
        "status": 200,
        "completion_text": "The product 19 x 23 is 437, so exp5124 telemetry is live.",
        "logprob_ready": True,
        "top_logprob_ready": True,
        "confidence_ready": False,
        "telemetry_signal": "top_logprobs",
        "evidence": {
            "token_logprob_count": 3,
            "top_logprob_row_count": 2,
            "token_logprobs": [-0.11, -0.22, -0.33],
            "top_logprobs": [{" The": -0.11, " A": -1.3}, {" product": -0.22}],
            "raw_response_keys": ["content", "completion_probabilities"],
        },
        "error": None,
    }


def _cache_sample(row_index: int, endpoint: str, selected_model: dict[str, Any]) -> dict[str, Any]:
    del selected_model
    sample = _ready_sample(endpoint, 5.0)
    sample["completion_text"] = f"cache row {row_index}: 19*23=437"
    return sample


def _clean_adversarial(_path: Path) -> dict[str, Any]:
    return {"flags": [], "summary": {"critical_count": 0}}


def test_req_infer_sota_029_spec_declares_exp5124_contract() -> None:
    """REQ-INFER-SOTA-029: OpenSpec anchors exp5124 fields and scenarios."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    source = Path(mod.__file__).read_text(encoding="utf-8")

    for marker in (
        "REQ-INFER-SOTA-029",
        "SCENARIO-INFER-SOTA-029-CLEAN",
        "SCENARIO-INFER-SOTA-029-BLOCKED",
        mod.EXPERIMENT_ID,
        mod.MILESTONE,
        mod.RESULT_RELATIVE_PATH,
        QWEN,
        GEMMA31,
        GEMMA26,
        "cached_sota_pair()",
        "duration_floor_evidence",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in mod.FIELD_PRINCIPLES
    assert "AutoTokenizer" not in source


def test_scenario_infer_sota_029_blocked_records_cached_pair_failure(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-029-BLOCKED: missing cached pair blocks the clean gate."""

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        cache_path=tmp_path / mod.CACHE_RELATIVE_PATH,
        model_resolver=lambda hf_id, preferred_quant: None,
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        precondition_probe=lambda root, env: _preconditions(),
        endpoint_probe=lambda endpoints, timeout_s: pytest.fail("endpoint probe must not run"),
        endpoint_sample=lambda endpoint, timeout_s: pytest.fail("sample must not run"),
        cache_sample=lambda row_index, endpoint, selected_model: pytest.fail("cache must not run"),
        server_finder=_server_unavailable,
        free_port=lambda host: _free_port(),
        adversarial_verify=_clean_adversarial,
        now=iter([10.0, 11.25]).__next__,
        duration_floor_s=60.0,
        tests_run=[{"command": "pytest tests/python/test_experiment_5124_clean_sota_runtime_provenance.py", "status": "passed"}],
        write=True,
    )

    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["honest_verdict"] == "blocked_clean_sota_runtime_provenance_cached_pair_unavailable"
    assert artifact["inference_substrate"] == "local_sota_gguf_llamacpp_runtime_or_blocked"
    assert artifact["cached_sota_pair_attempted"] is True
    assert artifact["cache_ready"] is False
    assert artifact["sota_runtime_clean"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["completion_proof"]["ready"] is False
    assert artifact["logprob_proof"]["ready"] is False
    assert artifact["cache_receipts"]["ready"] is False
    assert artifact["duration_floor_evidence"]["completed"] is False
    assert artifact["root_cause_tree"]["cached_sota_pair"]["present"] is True
    assert artifact["adversarial_verify_passed"] is True
    assert artifact["flagged_adversarial"] is False
    assert not (tmp_path / mod.CACHE_RELATIVE_PATH).exists()
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_infer_sota_029_clean_records_runtime_cache_and_duration_evidence(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-029-CLEAN: live logprobs plus cache readback open the gate."""

    paths = _write_models(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        cache_path=tmp_path / mod.CACHE_RELATIVE_PATH,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        precondition_probe=lambda root, env: _preconditions(),
        endpoint_probe=_ready_probe,
        endpoint_sample=_ready_sample,
        cache_sample=_cache_sample,
        server_finder=_server_unavailable,
        free_port=lambda host: _free_port(),
        adversarial_verify=_clean_adversarial,
        endpoints=["http://ready.test", "http://ready.test/"],
        now=iter([100.0, 165.0]).__next__,
        duration_floor_s=60.0,
        tests_run=[{"command": "pytest tests/python -q", "status": "passed"}],
        write=True,
    )

    assert artifact["honest_verdict"] == "success_clean_sota_runtime_provenance_ready"
    assert artifact["completion_proof"]["ready"] is True
    assert artifact["completion_proof"]["prompt"] == mod.DEFAULT_PROMPT
    assert artifact["logprob_proof"]["ready"] is True
    assert artifact["logprob_proof"]["token_logprob_count"] == 3
    assert artifact["cache_ready"] is True
    assert artifact["cache_receipts"]["ready"] is True
    assert artifact["cache_receipts"]["rows_written"] == 1
    assert artifact["cache_receipts"]["rows_read"] == 1
    assert artifact["cache_receipts"]["readback_matches"] is True
    assert artifact["duration_s"] == pytest.approx(65.0)
    assert artifact["endpoint_lifetime_s"] == pytest.approx(65.0)
    assert artifact["duration_floor_evidence"]["completed"] is True
    assert artifact["duration_floor_evidence"]["reason"] == "measured_wall_clock_duration_met_floor"
    assert artifact["request_response_transcript"]["completion_request"]["endpoint"] == (
        "http://ready.test/completion"
    )
    assert artifact["request_response_transcript"]["completion_response"]["text"].startswith(
        "The product 19 x 23 is 437"
    )
    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == [QWEN, GEMMA31, GEMMA26]
    assert artifact["MODEL_SPECS"][0]["model_path"] == paths[QWEN]
    assert artifact["MODEL_SPECS"][2]["model_path"] == paths[GEMMA26]
    assert artifact["gguf_paths"][GEMMA31] == paths[GEMMA31]
    assert artifact["adversarial_verify_passed"] is True
    assert artifact["sota_runtime_clean"] is True
    assert artifact["root_cause_tree"]["summary"] == "clean_runtime_provenance"
    assert mod.artifact_schema_errors(artifact) == []

    cache_rows = mod.read_jsonl_rows(tmp_path / mod.CACHE_RELATIVE_PATH)
    assert len(cache_rows) == 1
    assert cache_rows[0]["schema"] == mod.CACHE_ROW_SCHEMA
    assert cache_rows[0]["token_logprob_count"] == 3
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact


def test_req_infer_sota_029_adversarial_critical_flag_blocks_clean_gate(
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-029: adversarial verification is part of the clean gate."""

    paths = _write_models(tmp_path)

    def flagged(_path: Path) -> dict[str, Any]:
        return {
            "flags": [
                {
                    "kind": "DURATION_TOO_SHORT",
                    "severity": "critical",
                    "detail": "test critical flag",
                }
            ]
        }

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        cache_path=tmp_path / mod.CACHE_RELATIVE_PATH,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        precondition_probe=lambda root, env: _preconditions(),
        endpoint_probe=_ready_probe,
        endpoint_sample=_ready_sample,
        cache_sample=_cache_sample,
        server_finder=_server_unavailable,
        free_port=lambda host: _free_port(),
        adversarial_verify=flagged,
        endpoints=["http://ready.test"],
        now=iter([100.0, 165.0]).__next__,
        duration_floor_s=60.0,
        tests_run=[],
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_clean_sota_runtime_provenance_adversarial_flag"
    assert artifact["adversarial_verify_passed"] is False
    assert artifact["flagged_adversarial"] is True
    assert artifact["sota_runtime_clean"] is False
    assert artifact["cache_ready"] is False
    assert artifact["root_cause_tree"]["adversarial_verify"]["present"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_req_infer_sota_029_committed_artifact_is_schema_valid() -> None:
    """REQ-INFER-SOTA-029: the checked-in deliverable satisfies the exp5124 schema."""

    artifact_path = REPO / mod.RESULT_RELATIVE_PATH
    assert artifact_path.exists()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["inference_substrate"] == "local_sota_gguf_llamacpp_runtime_or_blocked"
    assert artifact["conductor_modified"] is False
    assert artifact["sota_runtime_clean"] is (
        bool(artifact["completion_proof"]["ready"])
        and bool(artifact["logprob_proof"]["ready"])
        and bool(artifact["cache_ready"])
        and bool(artifact["adversarial_verify_passed"])
    )
