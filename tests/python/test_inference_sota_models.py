"""Tests for carnot.inference.sota_models — the mandated SOTA GGUF registry.

Covers REQ-INFER-SOTA-001 (registry exists), REQ-INFER-SOTA-002 (helpers
return the expected models), and REQ-INFER-SOTA-003 (every entry has the
fields downstream code needs to drive llama.cpp loading).

Target: 100% line coverage per CLAUDE.md.
"""

from __future__ import annotations

import pytest

from carnot.inference.sota_models import (
    SOTA_GGUF_MODELS,
    SotaModelSpec,
    default_pair,
    flagship_dense,
    flagship_moe,
)


# SCENARIO-INFER-SOTA-001: registry contains the three mandated entries.
def test_registry_contains_three_mandated_models() -> None:
    assert len(SOTA_GGUF_MODELS) == 3
    hf_ids = {m["hf_id"] for m in SOTA_GGUF_MODELS}
    assert hf_ids == {
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
    }


# SCENARIO-INFER-SOTA-002: every record has the fields experiments read.
def test_every_record_has_required_fields() -> None:
    required = {
        "name", "hf_id", "role", "active_params_b",
        "total_params_b", "quantization", "min_vram_gb",
    }
    for record in SOTA_GGUF_MODELS:
        assert required <= set(record.keys())
        assert record["role"] in ("moe", "dense")
        assert record["active_params_b"] <= record["total_params_b"]
        assert record["min_vram_gb"] > 0


# SCENARIO-INFER-SOTA-003: flagship MoE is Qwen 3.6 35B A3B.
def test_flagship_moe_returns_qwen() -> None:
    m = flagship_moe()
    assert m["hf_id"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert m["role"] == "moe"


# SCENARIO-INFER-SOTA-004: flagship dense is Gemma 4 31B it.
def test_flagship_dense_returns_gemma31() -> None:
    m = flagship_dense()
    assert m["hf_id"] == "unsloth/gemma-4-31B-it-GGUF"
    assert m["role"] == "dense"


# SCENARIO-INFER-SOTA-005: default_pair shape is drop-in for MODEL_SPECS.
def test_default_pair_shape_matches_experiment_specs() -> None:
    pair = default_pair()
    assert len(pair) == 2
    for entry in pair:
        assert set(entry.keys()) == {"name", "hf_id", "gpu"}
    # Flagship MoE on GPU 0 per convention.
    assert pair[0]["hf_id"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert pair[0]["gpu"] == 0
    assert pair[1]["gpu"] == 1


# SCENARIO-INFER-SOTA-006: custom GPU indices propagate.
def test_default_pair_respects_custom_gpu_indices() -> None:
    pair = default_pair(gpu_indices=(2, 3))
    assert pair[0]["gpu"] == 2
    assert pair[1]["gpu"] == 3


# SCENARIO-INFER-SOTA-007: TypedDict is importable and usable.
def test_typed_dict_is_importable() -> None:
    # The type should at least be constructable with literal dicts.
    spec: SotaModelSpec = {
        "name": "test",
        "hf_id": "test/test-GGUF",
        "role": "moe",
        "active_params_b": 1.0,
        "total_params_b": 1.0,
        "quantization": "Q4_K_M",
        "min_vram_gb": 4,
    }
    assert spec["name"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
