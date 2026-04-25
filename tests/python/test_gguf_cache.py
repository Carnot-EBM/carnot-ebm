"""Tests for carnot.pipeline.gguf_cache.

Traces to REQ-PIPELINE-030, SCENARIO-PIPELINE-040.
"""

import os

import pytest

from carnot.pipeline.gguf_cache import (
    GGUFCacheConfig,
    GGUFCacheResolver,
    GGUFModelNotFoundError,
    resolve_gguf_path,
)


def test_config_defaults() -> None:
    """GGUFCacheConfig must have the documented defaults.

    Spec: REQ-PIPELINE-030
    """
    cfg = GGUFCacheConfig()
    assert cfg.cache_dir == "models/"
    assert cfg.default_quantization == "Q4_K_M"
    assert cfg.timeout_s == 30


def test_resolve_with_org_slash_name(tmp_path: pytest.TempPathFactory) -> None:
    """Org/name model ID must map to org_name-quant.gguf inside cache_dir.

    The slash separator between org and model name is converted to ``_``
    so the result is a flat filename valid on any filesystem.

    Spec: REQ-PIPELINE-030, SCENARIO-PIPELINE-040
    """
    model_id = "unsloth/Qwen3.6-35B-A3B-GGUF"
    expected_filename = "unsloth_Qwen3.6-35B-A3B-GGUF-Q4_K_M.gguf"

    # Create the file so resolve() does not raise.
    model_file = tmp_path / expected_filename
    model_file.write_bytes(b"")

    cfg = GGUFCacheConfig(cache_dir=str(tmp_path))
    resolver = GGUFCacheResolver(cfg)
    result = resolver.resolve(model_id)
    assert result.endswith(expected_filename)


def test_resolve_file_found(tmp_path: pytest.TempPathFactory) -> None:
    """resolve() returns the path when the .gguf file exists on disk.

    Spec: REQ-PIPELINE-030
    """
    model_id = "unsloth/Qwen3.6-35B-A3B-GGUF"
    filename = "unsloth_Qwen3.6-35B-A3B-GGUF-Q4_K_M.gguf"
    (tmp_path / filename).write_bytes(b"fake-gguf")

    cfg = GGUFCacheConfig(cache_dir=str(tmp_path))
    resolver = GGUFCacheResolver(cfg)
    path = resolver.resolve(model_id)
    assert os.path.exists(path)


def test_resolve_file_not_found(tmp_path: pytest.TempPathFactory) -> None:
    """resolve() must raise GGUFModelNotFoundError when the file is absent.

    The error details must include ``expected_path`` so the caller can
    surface a meaningful message to the user.

    Spec: REQ-PIPELINE-030, SCENARIO-PIPELINE-040
    """
    model_id = "unsloth/Qwen3.6-35B-A3B-GGUF"
    cfg = GGUFCacheConfig(cache_dir=str(tmp_path))
    resolver = GGUFCacheResolver(cfg)

    with pytest.raises(GGUFModelNotFoundError) as exc_info:
        resolver.resolve(model_id)

    assert "expected_path" in exc_info.value.details
    assert exc_info.value.details["model_id"] == model_id


def test_is_cached_true(tmp_path: pytest.TempPathFactory) -> None:
    """is_cached() returns True when the resolved file exists.

    Spec: REQ-PIPELINE-030
    """
    model_id = "unsloth/gemma-4-31B-it-GGUF"
    filename = "unsloth_gemma-4-31B-it-GGUF-Q4_K_M.gguf"
    (tmp_path / filename).write_bytes(b"")

    cfg = GGUFCacheConfig(cache_dir=str(tmp_path))
    assert GGUFCacheResolver(cfg).is_cached(model_id) is True


def test_is_cached_false(tmp_path: pytest.TempPathFactory) -> None:
    """is_cached() returns False when the resolved file is absent.

    Spec: REQ-PIPELINE-030
    """
    cfg = GGUFCacheConfig(cache_dir=str(tmp_path))
    assert GGUFCacheResolver(cfg).is_cached("unsloth/gemma-4-31B-it-GGUF") is False


def test_convenience_function(tmp_path: pytest.TempPathFactory) -> None:
    """resolve_gguf_path() must return the same path as resolver.resolve().

    Spec: REQ-PIPELINE-030
    """
    model_id = "unsloth/gemma-4-26B-A4B-it-GGUF"
    filename = "unsloth_gemma-4-26B-A4B-it-GGUF-Q4_K_M.gguf"
    (tmp_path / filename).write_bytes(b"")

    cfg = GGUFCacheConfig(cache_dir=str(tmp_path))
    expected = GGUFCacheResolver(cfg).resolve(model_id)
    result = resolve_gguf_path(model_id, cache_dir=str(tmp_path))
    assert result == expected
