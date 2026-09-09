"""Tests for the current local GGUF registry and legacy comparator path.

Spec refs: REQ-INFER-SOTA-7157 and SCENARIO-INFER-SOTA-7157-*.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import carnot.inference.sota_models as sota_models


QWEN38_ID = "unsloth/Qwen3.8-27B-GGUF"
QWEN36_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
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


# REQ-INFER-SOTA-7157 / SCENARIO-INFER-SOTA-7157-CURRENT.
def test_registry_contains_one_current_qwen38_model() -> None:
    assert sota_models.SOTA_GGUF_MODELS == [sota_models.current_model()]
    assert sota_models.current_model() == {
        "name": "Qwen3.8-27B",
        "hf_id": QWEN38_ID,
        "role": "dense",
        "active_params_b": 27.0,
        "total_params_b": 27.0,
        "quantization": "Q4_K_M",
        "min_vram_gb": 18,
        "mandate_status": "current_headline",
    }


# REQ-INFER-SOTA-7157 / SCENARIO-INFER-SOTA-7157-COMPARATORS.
def test_old_models_are_named_legacy_comparators() -> None:
    assert [row["hf_id"] for row in sota_models.LEGACY_COMPARATOR_GGUF_MODELS] == [
        QWEN36_ID,
        GEMMA26_ID,
        GEMMA31_ID,
    ]
    assert all(
        row["mandate_status"] == "legacy_comparator"
        for row in sota_models.LEGACY_COMPARATOR_GGUF_MODELS
    )


# REQ-INFER-SOTA-7157 / SCENARIO-INFER-SOTA-7157-COMPARATORS.
def test_explicit_legacy_helpers_keep_old_identities() -> None:
    assert sota_models.flagship_moe()["hf_id"] == QWEN36_ID
    assert sota_models.flagship_dense()["hf_id"] == GEMMA31_ID
    assert sota_models.flagship_moe()["mandate_status"] == "legacy_comparator"
    assert sota_models.flagship_dense()["mandate_status"] == "legacy_comparator"


# REQ-INFER-SOTA-7157 / SCENARIO-INFER-SOTA-7157-CURRENT.
def test_default_pair_labels_current_and_comparator() -> None:
    pair = sota_models.default_pair(gpu_indices=(4, 5))
    assert [row["hf_id"] for row in pair] == [QWEN38_ID, QWEN36_ID]
    assert [row["gpu"] for row in pair] == [4, 5]
    assert [row["selection_role"] for row in pair] == [
        "current_headline",
        "legacy_comparator",
    ]


# REQ-INFER-SOTA-7157 / SCENARIO-INFER-SOTA-7157-CURRENT.
def test_cached_current_model_resolves_only_qwen38(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cached_ids(monkeypatch, {QWEN38_ID, QWEN36_ID})

    current = sota_models.cached_current_model(gpu_index=3)

    assert current == {
        "name": "Qwen3.8-27B",
        "hf_id": QWEN38_ID,
        "gpu": 3,
        "model_path": "/cache/Qwen3.8-27B-Q4_K_M.gguf",
        "selection_role": "current_headline",
    }
    assert calls == [(QWEN38_ID, "Q4_K_M")]


# REQ-INFER-SOTA-7157 / SCENARIO-INFER-SOTA-7157-CURRENT.
def test_cached_current_model_returns_none_on_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cached_ids(monkeypatch, {QWEN36_ID})

    assert sota_models.cached_current_model() is None
    assert calls == [(QWEN38_ID, "Q4_K_M")]


# REQ-INFER-SOTA-7157 / SCENARIO-INFER-SOTA-7157-DEFAULT-PAIR.
def test_default_cached_pair_exposes_qwen38_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cached_ids(monkeypatch, {QWEN38_ID, GEMMA26_ID})

    pair = sota_models.cached_sota_pair(gpu_indices=(6, 7))

    assert pair is not None
    assert [row["hf_id"] for row in pair] == [QWEN38_ID, GEMMA26_ID]
    assert [row["selection_role"] for row in pair] == [
        "current_headline",
        "legacy_comparator",
    ]
    assert calls == [
        (QWEN38_ID, "Q4_K_M"),
        (QWEN36_ID, "Q4_K_M"),
        (GEMMA26_ID, "Q4_K_M"),
        (GEMMA31_ID, "Q4_K_M"),
    ]


# REQ-INFER-SOTA-7157 / SCENARIO-INFER-SOTA-7157-DEFAULT-PAIR.
def test_default_cached_pair_requires_current_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cached_ids(monkeypatch, {QWEN36_ID, GEMMA26_ID, GEMMA31_ID})

    assert sota_models.cached_sota_pair() is None


# REQ-INFER-SOTA-7157 / SCENARIO-INFER-SOTA-7157-COMPARATORS.
def test_explicit_model_indices_keep_legacy_pair_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_cached_ids(monkeypatch, {QWEN36_ID, GEMMA26_ID})

    pair = sota_models.cached_sota_pair(model_indices=(0, 1))

    assert pair is not None
    assert [row["hf_id"] for row in pair] == [QWEN36_ID, GEMMA26_ID]
    assert all(row["selection_role"] == "legacy_comparator" for row in pair)
    assert calls == [(QWEN36_ID, "Q4_K_M"), (GEMMA26_ID, "Q4_K_M")]


# REQ-INFER-SOTA-7157: projector files cannot satisfy the model cache gate.
def test_resolve_cached_gguf_ignores_newer_projector_snapshot(tmp_path: Path) -> None:
    cache = tmp_path / "hub"
    model_dir = cache / "models--unsloth--Qwen3.8-27B-GGUF" / "snapshots"
    old = model_dir / "old-revision"
    new = model_dir / "new-revision"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    language_model = old / "Qwen3.8-27B-Q4_K_M.gguf"
    projector = new / "mmproj-F16.gguf"
    language_model.write_bytes(b"GGUF language model placeholder")
    projector.write_bytes(b"GGUF projector placeholder")

    resolved = sota_models.resolve_cached_gguf(QWEN38_ID, cache_root=str(cache))

    assert resolved == str(language_model)
