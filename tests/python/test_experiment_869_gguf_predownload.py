"""Tests for Exp 869 — GGUFCacheResolver.pre_download_and_verify() and resolve_or_download().

All network calls are mocked so these tests run offline.

Traces to REQ-INFRA-073, SCENARIO-INFRA-082.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.gguf_cache import (
    GGUFCacheConfig,
    GGUFCacheResolver,
    GGUFModelNotFoundError,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_resolver(tmp_path: Path) -> GGUFCacheResolver:
    """Return a resolver pointing at tmp_path as cache_dir."""
    return GGUFCacheResolver(GGUFCacheConfig(cache_dir=str(tmp_path)))


# ── pre_download_and_verify — success path ────────────────────────────────────


def test_pre_download_success(tmp_path: Path) -> None:
    """pre_download_and_verify() returns success=True and size_mb > 0 when download works.

    Spec: REQ-INFRA-073, SCENARIO-INFRA-082
    """
    fake_file = tmp_path / "qwen3.5-0.8b-q4_k_m.gguf"
    # Write > 1 MB so size_mb rounds to > 0.0 with 2 decimal places.
    fake_file.write_bytes(b"x" * (1024 * 1024 + 1))

    resolver = _make_resolver(tmp_path)
    with patch("huggingface_hub.hf_hub_download", return_value=str(fake_file)):
        result = resolver.pre_download_and_verify(
            "Qwen/Qwen3.5-0.8B-GGUF", "qwen3.5-0.8b-q4_k_m.gguf", str(tmp_path)
        )

    assert result["success"] is True
    assert result["size_mb"] is not None
    assert result["size_mb"] > 0
    assert result["path"] == str(fake_file)
    assert result["error"] is None
    # download_tested is set after first successful call.
    assert resolver.download_tested is True


def test_pre_download_sets_download_tested(tmp_path: Path) -> None:
    """download_tested attribute becomes True after a successful pre_download_and_verify().

    Spec: REQ-INFRA-073
    """
    fake_file = tmp_path / "model.gguf"
    fake_file.write_bytes(b"data")

    resolver = _make_resolver(tmp_path)
    assert resolver.download_tested is False  # starts False

    with patch("huggingface_hub.hf_hub_download", return_value=str(fake_file)):
        resolver.pre_download_and_verify("Org/Repo", "model.gguf", str(tmp_path))

    assert resolver.download_tested is True


# ── pre_download_and_verify — failure paths ───────────────────────────────────


def test_pre_download_hf_hub_missing(tmp_path: Path) -> None:
    """pre_download_and_verify() returns success=False when huggingface_hub is missing.

    Spec: REQ-INFRA-073
    """
    resolver = _make_resolver(tmp_path)
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("No module named 'huggingface_hub'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        result = resolver.pre_download_and_verify("Org/Repo", "model.gguf", str(tmp_path))

    assert result["success"] is False
    assert result["error"] is not None
    assert "huggingface_hub" in result["error"]
    assert resolver.download_tested is False


def test_pre_download_oserror(tmp_path: Path) -> None:
    """pre_download_and_verify() returns success=False when hf_hub_download raises OSError.

    Spec: REQ-INFRA-073
    """
    resolver = _make_resolver(tmp_path)
    with patch("huggingface_hub.hf_hub_download", side_effect=OSError("connection refused")):
        result = resolver.pre_download_and_verify(
            "Qwen/Qwen3.5-0.8B-GGUF", "qwen3.5-0.8b-q4_k_m.gguf", str(tmp_path)
        )

    assert result["success"] is False
    assert "connection refused" in result["error"]
    assert result["path"] is None
    assert result["size_mb"] is None


def test_pre_download_file_missing_after_download(tmp_path: Path) -> None:
    """pre_download_and_verify() returns success=False when hf_hub_download returns a path that doesn't exist.

    This guards against hf_hub returning a symlink path after a network failure.

    Spec: REQ-INFRA-073
    """
    nonexistent = str(tmp_path / "ghost.gguf")
    resolver = _make_resolver(tmp_path)
    with patch("huggingface_hub.hf_hub_download", return_value=nonexistent):
        result = resolver.pre_download_and_verify("Org/Repo", "ghost.gguf", str(tmp_path))

    assert result["success"] is False
    assert "does not exist" in result["error"]


def test_pre_download_zero_byte_file(tmp_path: Path) -> None:
    """pre_download_and_verify() returns success=False for a 0-byte file.

    Spec: REQ-INFRA-073
    """
    empty_file = tmp_path / "empty.gguf"
    empty_file.write_bytes(b"")

    resolver = _make_resolver(tmp_path)
    with patch("huggingface_hub.hf_hub_download", return_value=str(empty_file)):
        result = resolver.pre_download_and_verify("Org/Repo", "empty.gguf", str(tmp_path))

    assert result["success"] is False
    assert result["size_mb"] == 0.0
    assert "0 bytes" in result["error"]


# ── resolve_or_download ────────────────────────────────────────────────────────


def test_resolve_or_download_uses_cache_when_present(tmp_path: Path) -> None:
    """resolve_or_download() returns cached path without downloading when file is in cache_dir.

    Spec: REQ-INFRA-073
    """
    model_id = "unsloth/Qwen3.6-35B-A3B-GGUF"
    filename = "unsloth_Qwen3.6-35B-A3B-GGUF-Q4_K_M.gguf"
    cached_file = tmp_path / filename
    cached_file.write_bytes(b"cached-model-data")

    resolver = _make_resolver(tmp_path)
    # hf_hub_download should NOT be called.
    with patch("huggingface_hub.hf_hub_download") as mock_dl:
        path = resolver.resolve_or_download(model_id, filename, str(tmp_path))
        mock_dl.assert_not_called()

    assert Path(path).exists()


def test_resolve_or_download_falls_back_to_download(tmp_path: Path) -> None:
    """resolve_or_download() calls pre_download_and_verify() when file is not cached.

    Spec: REQ-INFRA-073
    """
    hf_repo = "Qwen/Qwen3.5-0.8B-GGUF"
    filename = "qwen3.5-0.8b-q4_k_m.gguf"

    fake_file = tmp_path / filename
    fake_file.write_bytes(b"model-bytes" * 100)

    resolver = _make_resolver(tmp_path)
    with patch("huggingface_hub.hf_hub_download", return_value=str(fake_file)):
        path = resolver.resolve_or_download(hf_repo, filename, str(tmp_path))

    assert Path(path).exists()


def test_resolve_or_download_raises_on_download_failure(tmp_path: Path) -> None:
    """resolve_or_download() raises FileNotFoundError when download fails.

    Spec: REQ-INFRA-073
    """
    resolver = _make_resolver(tmp_path)
    with patch(
        "huggingface_hub.hf_hub_download",
        side_effect=OSError("network error"),
    ):
        with pytest.raises(FileNotFoundError, match="network error"):
            resolver.resolve_or_download("Org/Repo", "missing.gguf", str(tmp_path))


def test_resolve_or_download_checks_dest_dir_directly(tmp_path: Path) -> None:
    """resolve_or_download() finds file in dest_dir even when not in cache_dir config.

    Spec: REQ-INFRA-073
    """
    hf_repo = "Qwen/Qwen3.5-0.8B-GGUF"
    filename = "qwen3.5-0.8b-q4_k_m.gguf"

    # Put file in dest_dir but NOT in the resolver's cache_dir.
    dest_dir = tmp_path / "alt_dest"
    dest_dir.mkdir()
    dest_file = dest_dir / filename
    dest_file.write_bytes(b"model" * 200)

    # cache_dir points somewhere else — file is absent there.
    cache_dir = tmp_path / "empty_cache"
    cache_dir.mkdir()
    resolver = GGUFCacheResolver(GGUFCacheConfig(cache_dir=str(cache_dir)))

    with patch("huggingface_hub.hf_hub_download") as mock_dl:
        path = resolver.resolve_or_download(hf_repo, filename, str(dest_dir))
        mock_dl.assert_not_called()

    assert Path(path).exists()
