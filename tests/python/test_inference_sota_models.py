"""Tests for carnot.inference.sota_models — the mandated SOTA GGUF registry.

Covers REQ-INFER-SOTA-001 (registry exists), REQ-INFER-SOTA-002 (helpers
return the expected models), and REQ-INFER-SOTA-003 (every entry has the
fields downstream code needs to drive llama.cpp loading).

Target: 100% line coverage per CLAUDE.md.
Spec: REQ-INFRA-073
"""

from __future__ import annotations

import pytest
import carnot.inference.sota_models as sota_models
from carnot.inference.sota_models import (
    SOTA_GGUF_MODELS,
    SotaModelSpec,
    default_pair,
    flagship_dense,
    flagship_moe,
)


QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA26_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
GEMMA31_ID = "unsloth/gemma-4-31B-it-GGUF"


def _patch_cached_ids(
    monkeypatch: pytest.MonkeyPatch, cached_ids: set[str]
) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def fake_resolver(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        calls.append((hf_id, preferred_quant))
        if hf_id not in cached_ids:
            return None
        filename = hf_id.split("/", 1)[-1].removesuffix("-GGUF")
        return f"/cache/{filename}-{preferred_quant}.gguf"

    monkeypatch.setattr(sota_models, "resolve_cached_gguf", fake_resolver)
    return calls


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
        "name",
        "hf_id",
        "role",
        "active_params_b",
        "total_params_b",
        "quantization",
        "min_vram_gb",
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


# REQ-INFER-SOTA-005 / SCENARIO-INFER-SOTA-005-001:
# any two cached mandated GGUFs are sufficient for a loadable pair.
def test_cached_sota_pair_returns_two_cached_mandated_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cached_ids(monkeypatch, {QWEN_ID, GEMMA26_ID})

    pair = sota_models.cached_sota_pair(gpu_indices=(4, 5), preferred_quant="Q4_K_M")

    assert pair is not None
    assert [entry["hf_id"] for entry in pair] == [QWEN_ID, GEMMA26_ID]
    assert [entry["gpu"] for entry in pair] == [4, 5]
    assert all(entry["model_path"].endswith("Q4_K_M.gguf") for entry in pair)
    assert calls == [
        (QWEN_ID, "Q4_K_M"),
        (GEMMA26_ID, "Q4_K_M"),
        (GEMMA31_ID, "Q4_K_M"),
    ]


# REQ-INFER-SOTA-005 / SCENARIO-INFER-SOTA-005-002:
# one cached mandated GGUF is not enough for a headline pair.
def test_cached_sota_pair_returns_none_for_one_cached_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cached_ids(monkeypatch, {QWEN_ID})

    pair = sota_models.cached_sota_pair()

    assert pair is None


# REQ-INFER-SOTA-005 / SCENARIO-INFER-SOTA-005-003:
# no cached mandated GGUFs blocks the cached pair path.
def test_cached_sota_pair_returns_none_for_no_cached_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cached_ids(monkeypatch, set())

    pair = sota_models.cached_sota_pair()

    assert pair is None


# REQ-INFER-SOTA-005 / SCENARIO-INFER-SOTA-005-001:
# the missing third mandated GGUF is optional once two loadable specs exist.
def test_cached_sota_pair_treats_missing_third_model_as_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cached_ids(monkeypatch, {QWEN_ID, GEMMA31_ID})

    pair = sota_models.cached_sota_pair(gpu_indices=(6, 7), preferred_quant="Q4_K_M")
    missing_optional_models = sorted(
        {model["hf_id"] for model in SOTA_GGUF_MODELS} - {entry["hf_id"] for entry in pair or []}
    )

    assert pair is not None
    assert [entry["hf_id"] for entry in pair] == [QWEN_ID, GEMMA31_ID]
    assert [entry["gpu"] for entry in pair] == [6, 7]
    assert missing_optional_models == [GEMMA26_ID]


# REQ-VERIFY-6146-2 / SCENARIO-VERIFY-6146-GATE:
# projector-only GGUFs must not satisfy a headline language-model path.
def test_resolve_cached_gguf_ignores_newer_projector_snapshot(tmp_path) -> None:
    cache = tmp_path / "hub"
    model_dir = cache / "models--unsloth--gemma-4-31B-it-GGUF" / "snapshots"
    old = model_dir / "old-revision"
    new = model_dir / "new-revision"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    language_model = old / "gemma-4-31B-it-Q4_K_M.gguf"
    projector = new / "mmproj-F16.gguf"
    language_model.write_bytes(b"GGUF language model placeholder")
    projector.write_bytes(b"GGUF projector placeholder")

    resolved = sota_models.resolve_cached_gguf(GEMMA31_ID, cache_root=str(cache))

    assert resolved == str(language_model)


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
