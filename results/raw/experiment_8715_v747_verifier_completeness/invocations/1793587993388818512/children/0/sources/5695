"""GGUFCacheResolver — find and download GGUF model files for local inference.

**Why this module exists:**
    The HuggingFace Python API (hf_hub_download) has a fragile download path
    that fails with RepositoryNotFoundError even when the repo exists, due to
    filename case mismatches, endpoint selection, and token handling differences.
    After 11 consecutive failures across experiments 857–869, this module adds a
    CLI-based fallback using the ``hf`` / ``huggingface-cli`` binary, which runs
    in a separate OS process and uses its own HTTP stack — a fundamentally different
    code path that avoids the Python API's failure mode.

**Design:**
    - ``resolve()`` checks the local HuggingFace cache without network I/O.
    - ``cli_download()`` invokes the ``hf`` CLI binary via subprocess, which
      produces correct downloads even when the Python API returns 404.
    - ``resolve_with_cli_fallback()`` composes both: cache-first, then CLI.

**Spec:** REQ-INFRA-074 (CLI download fallback), SCENARIO-INFRA-083
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class GGUFCacheResolver:
    """Resolve GGUF model files from the local HuggingFace cache, with a CLI fallback.

    **What this class does:**
        It looks in the standard HuggingFace cache (``~/.cache/huggingface/hub``)
        for a GGUF file that was previously downloaded.  If the file is not cached,
        ``resolve_with_cli_fallback()`` downloads it using the ``hf`` or
        ``huggingface-cli`` command-line tool rather than the Python API, because
        the CLI tool has a separate HTTP stack that is more reliable for large
        binary files.

    **Why not use the Python API only:**
        The Python API routes through a requests session that applies case-sensitive
        filename matching and may pick the wrong CDN endpoint.  The CLI binary applies
        its own retry + redirect logic and handles GGUF blobs correctly in practice.
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        """Initialise the resolver, optionally pointing at a custom cache directory.

        Args:
            cache_dir: Path to the HuggingFace hub cache.  Defaults to
                       ``~/.cache/huggingface/hub`` (the standard HF cache location).
        """
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = (
                Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
            )

    def resolve(self, hf_repo: str, filename: str) -> Path | None:
        """Return the cached path for ``filename`` in ``hf_repo``, or None.

        **What it does:**
            Looks for the file in the standard HuggingFace snapshot cache layout
            without making any network requests.  Returns None rather than raising
            if the file is absent.

        **Why no network I/O here:**
            ``resolve()`` is intentionally read-only so callers can distinguish
            "file already downloaded" from "file needs downloading" and route
            accordingly (e.g. skip the CLI invocation when warm cache is present).

        Args:
            hf_repo: HuggingFace repo ID, e.g. ``"Qwen/Qwen3.5-0.8B-GGUF"``.
            filename: The specific file within the repo, e.g.
                      ``"Qwen3.5-0.8B-Q4_K_M.gguf"``.

        Returns:
            A ``pathlib.Path`` pointing to the cached file, or ``None`` if not found.
        """
        # HF cache layout: hub/models--<owner>--<name>/snapshots/<hash>/<filename>
        # We do a glob to avoid needing the exact snapshot hash.
        repo_slug = "models--" + hf_repo.replace("/", "--")
        repo_path = self.cache_dir / repo_slug
        if not repo_path.exists():
            return None

        for candidate in repo_path.glob(f"snapshots/*/{filename}"):
            if candidate.is_file():
                return candidate

        return None

    def cli_download(
        self,
        hf_repo: str,
        filename: str,
        dest_dir: str,
        timeout_s: int = 300,
    ) -> dict:
        """Download a GGUF file using the ``hf`` CLI binary via subprocess.

        **Why subprocess instead of the Python API:**
            The ``hf`` / ``huggingface-cli`` binary manages its own HTTP session,
            handles redirects differently, and retries on transient errors without
            the case-sensitivity issue that caused 11 consecutive Python-API failures
            (experiments 857–869, verdict: RepositoryNotFoundError even though the
            repo exists).

        **What it does:**
            1. Finds the ``hf`` or ``huggingface-cli`` binary on PATH.
            2. Runs: ``hf download <hf_repo> <filename> --local-dir <dest_dir>``
            3. After the subprocess exits, verifies the file actually landed on disk.
            4. Returns a result dict with ``success``, ``path``, ``size_mb``, or
               ``error`` depending on the outcome.

        Args:
            hf_repo: HuggingFace repo ID (e.g. ``"Qwen/Qwen3.5-0.8B-GGUF"``).
            filename: File to download within the repo.
            dest_dir: Local directory to download into.
            timeout_s: Subprocess wall-clock timeout in seconds.  Default 300 s
                       (5 minutes) is sufficient for most sub-5 GB GGUF files on a
                       normal connection; increase for very large files.

        Returns:
            ``{"success": True, "path": str, "size_mb": float}`` on success.
            ``{"success": False, "error": str}`` on any failure (CLI not found,
            non-zero returncode, or file absent after download).

        Spec: REQ-INFRA-074, SCENARIO-INFRA-083
        """
        hf_cmd = shutil.which("hf") or shutil.which("huggingface-cli")
        if hf_cmd is None:
            return {
                "success": False,
                "error": "hf CLI not found: install with 'pip install huggingface_hub[cli]' or 'apt install huggingface-hub'",
            }

        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)

        cmd = [hf_cmd, "download", hf_repo, filename, "--local-dir", str(dest_path)]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"hf CLI timed out after {timeout_s}s",
            }

        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr[:500] if result.stderr else result.stdout[:500],
            }

        # Verify the file actually landed on disk — the CLI can exit 0 and still
        # produce nothing if the filename is wrong.
        expected = dest_path / filename
        if not expected.exists():
            # Some CLI versions put files in a subdirectory; search shallowly.
            candidates = list(dest_path.rglob(filename))
            if not candidates:
                return {
                    "success": False,
                    "error": f"CLI exited 0 but {filename} not found under {dest_path}",
                }
            expected = candidates[0]

        size_mb = expected.stat().st_size / (1024 * 1024)
        return {
            "success": True,
            "path": str(expected),
            "size_mb": round(size_mb, 2),
        }

    def resolve_with_cli_fallback(
        self,
        hf_repo: str,
        filename: str,
        dest_dir: str,
        timeout_s: int = 300,
    ) -> Path:
        """Return a local path for ``filename``, downloading via CLI if not cached.

        **Lookup order:**
            1. Check the local HF cache (``resolve()``).  Zero network I/O.
            2. If not found, run the CLI downloader (``cli_download()``).
            3. If both fail, raise ``FileNotFoundError`` with a diagnostic message
               that names both failure modes so the caller can decide what to log.

        Args:
            hf_repo: HuggingFace repo ID.
            filename: File to resolve.
            dest_dir: Where the CLI should download if cache misses.
            timeout_s: Passed through to ``cli_download()``.

        Returns:
            ``pathlib.Path`` of the resolved file.

        Raises:
            FileNotFoundError: If neither the cache nor the CLI produced the file.

        Spec: REQ-INFRA-074
        """
        cached = self.resolve(hf_repo, filename)
        if cached is not None:
            return cached

        result = self.cli_download(hf_repo, filename, dest_dir, timeout_s=timeout_s)
        if result["success"]:
            return Path(result["path"])

        raise FileNotFoundError(
            f"Could not resolve {hf_repo}/{filename}. "
            f"Cache miss and CLI download failed: {result.get('error', 'unknown')}"
        )
