"""Tests for GGUFCacheResolver CLI download methods.

Spec: REQ-INFRA-074, SCENARIO-INFRA-083

These tests mock subprocess.run so no network I/O occurs.  Every test has at
least one assertion.  100% coverage of the code added in
python/carnot/resolvers/gguf_cache.py.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from python.carnot.resolvers.gguf_cache import GGUFCacheResolver


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def resolver(tmp_path: Path) -> GGUFCacheResolver:
    """GGUFCacheResolver pointing at a temporary cache directory."""
    return GGUFCacheResolver(cache_dir=str(tmp_path))


def _make_fake_snapshot(tmp_path: Path, repo: str, filename: str) -> Path:
    """Create a fake HF snapshot layout and plant a small file in it."""
    repo_slug = "models--" + repo.replace("/", "--")
    snap = tmp_path / repo_slug / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    f = snap / filename
    f.write_bytes(b"fake gguf content here")
    return f


# ---------------------------------------------------------------------------
# GGUFCacheResolver.resolve()
# ---------------------------------------------------------------------------


class TestResolve:
    """Spec: REQ-INFRA-074 — resolve() finds cached files without network I/O."""

    def test_returns_none_when_repo_absent(self, resolver: GGUFCacheResolver) -> None:
        """resolve() returns None when the repo directory does not exist in cache."""
        result = resolver.resolve("NoOrg/NonExistentRepo", "model.gguf")
        assert result is None

    def test_returns_path_when_file_cached(
        self, resolver: GGUFCacheResolver, tmp_path: Path
    ) -> None:
        """resolve() returns the cached Path when the snapshot file exists."""
        fake = _make_fake_snapshot(tmp_path, "Qwen/Qwen3.5-0.8B-GGUF", "model.gguf")
        result = resolver.resolve("Qwen/Qwen3.5-0.8B-GGUF", "model.gguf")
        assert result is not None
        assert result == fake
        assert result.is_file()

    def test_returns_none_when_wrong_filename(
        self, resolver: GGUFCacheResolver, tmp_path: Path
    ) -> None:
        """resolve() returns None when the repo exists but the filename doesn't match."""
        _make_fake_snapshot(tmp_path, "Qwen/Qwen3.5-0.8B-GGUF", "model.gguf")
        result = resolver.resolve("Qwen/Qwen3.5-0.8B-GGUF", "other.gguf")
        assert result is None


# ---------------------------------------------------------------------------
# GGUFCacheResolver.cli_download() — success path
# ---------------------------------------------------------------------------


class TestCliDownloadSuccess:
    """Spec: SCENARIO-INFRA-083 — CLI download returns success=True on valid repo."""

    def test_success_path_returns_dict_with_path_and_size(self, tmp_path: Path) -> None:
        """cli_download() returns success=True with path and size_mb when CLI exits 0."""
        dest = tmp_path / "dest"
        dest.mkdir()
        fake_file = dest / "model.gguf"
        fake_file.write_bytes(b"x" * 1024 * 512)  # 0.5 MB

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = ""

        resolver = GGUFCacheResolver()

        with (
            patch("shutil.which", return_value="/usr/bin/hf"),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = resolver.cli_download(
                "Qwen/Qwen3.5-0.8B-GGUF",
                "model.gguf",
                str(dest),
                timeout_s=30,
            )

        assert result["success"] is True
        assert "path" in result
        assert result["size_mb"] > 0

    def test_subprocess_called_with_correct_args(self, tmp_path: Path) -> None:
        """cli_download() calls subprocess.run with the expected hf download command."""
        dest = tmp_path / "dest"
        dest.mkdir()
        (dest / "model.gguf").write_bytes(b"data")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        resolver = GGUFCacheResolver()

        with (
            patch("shutil.which", return_value="/usr/bin/hf") as mock_which,
            patch("subprocess.run", return_value=mock_result) as mock_run,
        ):
            resolver.cli_download("MyOrg/MyRepo", "model.gguf", str(dest))

        mock_run.assert_called_once()
        args = mock_run.call_args
        cmd = args[0][0]
        assert cmd[0] == "/usr/bin/hf"
        assert "download" in cmd
        assert "MyOrg/MyRepo" in cmd
        assert "model.gguf" in cmd
        assert "--local-dir" in cmd

    def test_uses_huggingface_cli_when_hf_absent(self, tmp_path: Path) -> None:
        """cli_download() falls back to huggingface-cli when hf is not on PATH."""
        dest = tmp_path / "dest"
        dest.mkdir()
        (dest / "model.gguf").write_bytes(b"data")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        resolver = GGUFCacheResolver()

        def _which(name: str) -> str | None:
            return None if name == "hf" else "/usr/bin/huggingface-cli"

        with (
            patch("shutil.which", side_effect=_which),
            patch("subprocess.run", return_value=mock_result) as mock_run,
        ):
            result = resolver.cli_download("MyOrg/MyRepo", "model.gguf", str(dest))

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/huggingface-cli"
        assert result["success"] is True


# ---------------------------------------------------------------------------
# GGUFCacheResolver.cli_download() — failure paths
# ---------------------------------------------------------------------------


class TestCliDownloadFailure:
    """Spec: REQ-INFRA-074 — failure paths return error dict, never raise."""

    def test_cli_not_found_returns_error_dict(self) -> None:
        """cli_download() returns success=False with descriptive error when CLI absent."""
        resolver = GGUFCacheResolver()
        with patch("shutil.which", return_value=None):
            result = resolver.cli_download("MyOrg/Repo", "model.gguf", "/tmp/dest")

        assert result["success"] is False
        assert "error" in result
        assert "hf CLI not found" in result["error"]

    def test_nonzero_returncode_returns_error_dict(self, tmp_path: Path) -> None:
        """cli_download() returns success=False when the CLI exits non-zero."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Repository Not Found"
        mock_result.stdout = ""

        resolver = GGUFCacheResolver()
        with (
            patch("shutil.which", return_value="/usr/bin/hf"),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = resolver.cli_download("Bad/Repo", "model.gguf", str(tmp_path))

        assert result["success"] is False
        assert "Repository Not Found" in result["error"]

    def test_timeout_returns_error_dict(self, tmp_path: Path) -> None:
        """cli_download() returns success=False when subprocess times out."""
        resolver = GGUFCacheResolver()
        with (
            patch("shutil.which", return_value="/usr/bin/hf"),
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=[], timeout=5)),
        ):
            result = resolver.cli_download("Org/Repo", "model.gguf", str(tmp_path), timeout_s=5)

        assert result["success"] is False
        assert "timed out" in result["error"]

    def test_file_absent_after_exit_zero_returns_error(self, tmp_path: Path) -> None:
        """cli_download() returns success=False when CLI exits 0 but file is missing."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = ""

        resolver = GGUFCacheResolver()
        with (
            patch("shutil.which", return_value="/usr/bin/hf"),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = resolver.cli_download("Org/Repo", "phantom.gguf", str(tmp_path))

        assert result["success"] is False
        assert "phantom.gguf" in result["error"]

    def test_file_found_in_subdirectory(self, tmp_path: Path) -> None:
        """cli_download() finds file placed in a subdirectory by the CLI."""
        dest = tmp_path / "dest"
        (dest / "sub").mkdir(parents=True)
        (dest / "sub" / "model.gguf").write_bytes(b"data" * 100)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        resolver = GGUFCacheResolver()
        with (
            patch("shutil.which", return_value="/usr/bin/hf"),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = resolver.cli_download("Org/Repo", "model.gguf", str(dest))

        assert result["success"] is True


# ---------------------------------------------------------------------------
# GGUFCacheResolver.resolve_with_cli_fallback()
# ---------------------------------------------------------------------------


class TestResolveWithCliFallback:
    """Spec: REQ-INFRA-074 — cache-first then CLI fallback."""

    def test_returns_cached_path_without_cli(
        self, resolver: GGUFCacheResolver, tmp_path: Path
    ) -> None:
        """resolve_with_cli_fallback() returns cached path without invoking CLI."""
        fake = _make_fake_snapshot(tmp_path, "Qwen/Qwen3.5-0.8B-GGUF", "model.gguf")

        with patch.object(resolver, "cli_download") as mock_cli:
            result = resolver.resolve_with_cli_fallback(
                "Qwen/Qwen3.5-0.8B-GGUF", "model.gguf", str(tmp_path / "dest")
            )

        mock_cli.assert_not_called()
        assert result == fake

    def test_calls_cli_on_cache_miss(self, resolver: GGUFCacheResolver, tmp_path: Path) -> None:
        """resolve_with_cli_fallback() invokes cli_download() when cache misses."""
        dest = tmp_path / "dest"
        dest.mkdir()
        fake_file = dest / "model.gguf"
        fake_file.write_bytes(b"data")

        with patch.object(
            resolver,
            "cli_download",
            return_value={"success": True, "path": str(fake_file), "size_mb": 0.01},
        ) as mock_cli:
            result = resolver.resolve_with_cli_fallback("Org/Repo", "model.gguf", str(dest))

        mock_cli.assert_called_once()
        assert result == fake_file

    def test_raises_file_not_found_when_both_fail(
        self, resolver: GGUFCacheResolver, tmp_path: Path
    ) -> None:
        """resolve_with_cli_fallback() raises FileNotFoundError when cache and CLI both fail."""
        with patch.object(
            resolver,
            "cli_download",
            return_value={"success": False, "error": "Repository Not Found"},
        ):
            with pytest.raises(FileNotFoundError, match="Repository Not Found"):
                resolver.resolve_with_cli_fallback(
                    "Bad/Repo", "missing.gguf", str(tmp_path / "dest")
                )


# ---------------------------------------------------------------------------
# Default cache_dir behaviour
# ---------------------------------------------------------------------------


class TestDefaultCacheDir:
    """Spec: REQ-INFRA-074 — default cache dir uses HF_HOME or ~/.cache/huggingface/hub."""

    def test_default_uses_hf_home_env(self, tmp_path: Path) -> None:
        """GGUFCacheResolver() uses HF_HOME env var when set."""
        with patch.dict(os.environ, {"HF_HOME": str(tmp_path)}):
            r = GGUFCacheResolver()
        assert r.cache_dir == tmp_path / "hub"

    def test_default_falls_back_to_home(self, tmp_path: Path) -> None:
        """GGUFCacheResolver() falls back to ~/.cache/huggingface/hub when HF_HOME unset."""
        env = os.environ.copy()
        env.pop("HF_HOME", None)
        with patch.dict(os.environ, env, clear=True):
            r = GGUFCacheResolver()
        assert r.cache_dir == Path.home() / ".cache" / "huggingface" / "hub"
