"""Tests for Exp 1891 SOTA GGUF cache/runtime preflight.

Spec: REQ-INFER-SOTA-011,
      SCENARIO-INFER-SOTA-011-001
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting.sota_gguf_cache_runtime_preflight import (
    MODEL_SPECS,
    REQUIRED_ARTIFACT_FIELDS,
    build_preflight_artifact,
    run_experiment,
)

QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _gpu_probe(*, available: bool = True) -> dict[str, Any]:
    return {
        "cuda_available": available,
        "gpu_count": 2 if available else 0,
        "nvidia_smi_available": available,
        "gpus": [{"index": 0, "name": "RTX 3090"}] if available else [],
    }


def test_exp1891_partial_cache_smoke_success_separates_readiness(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-011 / SCENARIO-INFER-SOTA-011-001: partial cache is not all-cache readiness."""
    smoke_calls: list[dict[str, Any]] = []

    def resolver(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        assert preferred_quant == "Q4_K_M"
        return {GEMMA26: "/cache/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"}.get(hf_id)

    def smoke_probe(model: dict[str, Any]) -> dict[str, Any]:
        smoke_calls.append(model)
        return {
            "hf_id": model["hf_id"],
            "model_path": model["model_path"],
            "runtime_mode": "llama_cpp_subprocess_gpu",
            "truly_live": True,
            "usable_response": True,
            "blocker": None,
        }

    artifact = build_preflight_artifact(
        project_root=tmp_path,
        run_date="20260512",
        cache_resolver=resolver,
        cached_pair_fn=lambda **_: None,
        gpu_probe_fn=lambda: _gpu_probe(),
        smoke_probe_fn=smoke_probe,
        materialize_environ={},
        materializer_fn=lambda **_: pytest.fail("materialization must be opt-in"),
    )

    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == [QWEN, GEMMA31, GEMMA26]
    assert [row["hf_id"] for row in MODEL_SPECS] == [QWEN, GEMMA31, GEMMA26]
    assert artifact["status"] == "complete"
    assert artifact["cache_all_available"] is False
    assert artifact["cache_any_available"] is True
    assert artifact["missing_models"] == [QWEN, GEMMA31]
    assert artifact["runtime_smoke_ready"] is True
    assert artifact["runtime_backend"] == "llama_cpp_subprocess_gpu"
    assert artifact["models_used"] == [GEMMA26]
    assert artifact["gpu_telemetry_available"] is True
    assert artifact["model_count"] == 1
    assert artifact["parallel_model_count"] == 0
    assert artifact["materialization_attempted"] is False
    assert smoke_calls == [
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": GEMMA26,
            "role": "middle_moe_secondary",
            "preferred_quant": "Q4_K_M",
            "gpu": 0,
            "min_vram_gb": 16,
            "model_path": "/cache/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        }
    ]
    assert all(row["candidate_only"] is True for row in artifact["fallback_runtime_candidates"])
    assert artifact["honest_verdict"].startswith("complete:")


def test_exp1891_missing_cache_skips_runtime_and_materialization(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-011-001: missing mandated cache blocks without fake smoke."""
    artifact = build_preflight_artifact(
        project_root=tmp_path,
        run_date="20260512",
        cache_resolver=lambda *_args, **_kwargs: None,
        cached_pair_fn=lambda **_: None,
        gpu_probe_fn=lambda: _gpu_probe(available=False),
        smoke_probe_fn=lambda *_args, **_kwargs: pytest.fail("smoke requires a cached model"),
        materialize_environ={},
        materializer_fn=lambda **_: pytest.fail("materialization must be opt-in"),
    )

    assert artifact["cache_all_available"] is False
    assert artifact["cache_any_available"] is False
    assert artifact["missing_models"] == [QWEN, GEMMA31, GEMMA26]
    assert artifact["runtime_smoke_ready"] is False
    assert artifact["runtime_backend"] == "none_no_cached_mandated_model"
    assert artifact["models_used"] == []
    assert artifact["gpu_telemetry_available"] is False
    assert artifact["model_count"] == 0
    assert artifact["parallel_model_count"] == 0
    assert artifact["smoke_results"] == []
    assert artifact["materialization_skipped_reason"] == "CARNOT_ALLOW_SOTA_GGUF_MATERIALIZE not set"


def test_exp1891_all_cache_counts_cached_pair_but_smokes_cheapest_model(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-011: cached_sota_pair evidence is telemetry, not the smoke selector."""
    paths = {
        QWEN: "/cache/qwen.gguf",
        GEMMA31: "/cache/gemma31.gguf",
        GEMMA26: "/cache/gemma26.gguf",
    }
    smoke_calls: list[str] = []

    artifact = build_preflight_artifact(
        project_root=tmp_path,
        run_date="20260512",
        cache_resolver=lambda hf_id, **_: paths[hf_id],
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [
            {"name": "Qwen3.6-35B-A3B", "hf_id": QWEN, "gpu": gpu_indices[0], "model_path": paths[QWEN]},
            {"name": "Gemma4-31B-it", "hf_id": GEMMA31, "gpu": gpu_indices[1], "model_path": paths[GEMMA31]},
        ],
        gpu_probe_fn=lambda: _gpu_probe(),
        smoke_probe_fn=lambda model: smoke_calls.append(model["hf_id"])
        or {
            "hf_id": model["hf_id"],
            "runtime_mode": "llama_cpp_subprocess_gpu",
            "truly_live": True,
            "usable_response": True,
            "blocker": None,
        },
        materialize_environ={},
    )

    assert artifact["cache_all_available"] is True
    assert artifact["cache_any_available"] is True
    assert artifact["missing_models"] == []
    assert artifact["model_count"] == 3
    assert artifact["parallel_model_count"] == 2
    assert artifact["runtime_smoke_ready"] is True
    assert artifact["models_used"] == [GEMMA26]
    assert smoke_calls == [GEMMA26]


def test_exp1891_materialization_is_bounded_and_reinspected(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-011: materialization runs only under explicit safe configuration."""
    available: dict[str, str] = {}
    attempts: list[tuple[str, int]] = []

    def materializer(*, hf_id: str, preferred_quant: str, timeout_s: int) -> dict[str, Any]:
        assert preferred_quant == "Q4_K_M"
        attempts.append((hf_id, timeout_s))
        if hf_id == QWEN:
            available[hf_id] = "/cache/qwen-after-materialize.gguf"
            return {"success": True, "path": available[hf_id], "error": None}
        return {"success": False, "path": None, "error": "bounded materialization unavailable"}

    artifact = build_preflight_artifact(
        project_root=tmp_path,
        run_date="20260512",
        cache_resolver=lambda hf_id, **_: available.get(hf_id),
        cached_pair_fn=lambda **_: None,
        gpu_probe_fn=lambda: _gpu_probe(),
        smoke_probe_fn=lambda model: {
            "hf_id": model["hf_id"],
            "runtime_mode": "llama_cpp_subprocess_gpu",
            "truly_live": False,
            "usable_response": False,
            "blocker": "load failed",
        },
        materialize_environ={
            "CARNOT_ALLOW_SOTA_GGUF_MATERIALIZE": "1",
            "CARNOT_SOTA_GGUF_MATERIALIZE_TIMEOUT_S": "not-an-int",
        },
        materializer_fn=materializer,
    )

    assert attempts == [(QWEN, 120), (GEMMA31, 120), (GEMMA26, 120)]
    assert artifact["materialization_attempted"] is True
    assert artifact["cache_any_available"] is True
    assert artifact["cache_all_available"] is False
    assert artifact["missing_models"] == [GEMMA31, GEMMA26]
    assert artifact["model_count"] == 1
    assert artifact["runtime_smoke_ready"] is False
    assert artifact["models_used"] == []


def test_exp1891_run_experiment_writes_required_json(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-011: runner writes the terminal Exp 1891 JSON artifact."""
    output_path = tmp_path / "results" / "experiment_1891_sota_gguf_cache_runtime_preflight.json"

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260512",
        output_path=output_path,
        cache_resolver=lambda *_args, **_kwargs: None,
        cached_pair_fn=lambda **_: None,
        gpu_probe_fn=lambda: _gpu_probe(available=False),
        smoke_probe_fn=lambda *_args, **_kwargs: pytest.fail("smoke requires cache"),
        materialize_environ={},
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "complete"
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(written)
    assert written["tests_run"] == []
