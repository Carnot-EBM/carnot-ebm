#!/usr/bin/env python3
"""SOPS-based secret decryption helper for Carnot experiment scripts.

**Why this module exists:**
    Experiment scripts need access to secrets (e.g. HF_TOKEN for HuggingFace uploads)
    without ever committing plaintext credentials to the repository.  SOPS (Mozilla
    Secrets OPerationS) encrypts secret files at rest using age/PGP/KMS keys, and
    decrypts them at runtime via `sops --decrypt`.

    This module provides a single entry-point ``decrypt_secret(key)`` that:
      1. Looks for a .sops.yaml secrets file in the repository root.
      2. Calls ``sops --decrypt --extract`` to retrieve the specific key value.
      3. Falls back to the environment variable of the same name if SOPS fails.
      4. Returns None if both SOPS and env var are absent (caller handles gracefully).

    Callers MUST NOT log or print the returned value — treat it as a write-once secret.

**SOPS file format expected (.sops.yaml or secrets.enc.yaml):**
    The encrypted YAML file must contain a top-level key matching the ``key`` argument.
    Example plaintext (before SOPS encryption):
        HF_TOKEN: hf_xxxxxxxxxxxxxxxxxxxxxxxx

**Security guarantee:**
    The plaintext secret is never written to disk by this module.  It is held in
    memory only for the duration of the calling experiment and is not cached.

Spec: REQ-INFRA-062 (secrets must not be committed in plaintext)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent

# Ordered list of candidate SOPS-encrypted files to search for secrets.
# The first file that exists and successfully decrypts wins.
_SOPS_CANDIDATES = [
    _REPO_ROOT / "secrets.enc.yaml",
    _REPO_ROOT / ".sops.yaml",
    _REPO_ROOT / "secrets.yaml",
]


def decrypt_secret(key: str, *, timeout: int = 10) -> str | None:
    """Return the plaintext value of ``key`` from a SOPS-encrypted file, or None.

    Decryption strategy (first success wins):
      1. Try ``sops --decrypt --extract '["<key>"]' <secrets_file>`` for each
         candidate secrets file in ``_SOPS_CANDIDATES``.
      2. Fall back to ``os.environ.get(key)`` — allows CI/CD environments to
         inject secrets via environment variables without SOPS installed.
      3. Return None if both strategies fail (caller MUST handle this gracefully
         by writing a blocked artifact with ``hf_auth_blocked=True``).

    Parameters
    ----------
    key : str
        The top-level YAML key to extract (e.g. "HF_TOKEN").
    timeout : int
        Seconds before the sops subprocess is killed.  Keep low (10 s) so
        runaway gpg-agent calls don't block the experiment watchdog.

    Returns
    -------
    str | None
        Decrypted value, or None if unavailable.
    """
    for candidate in _SOPS_CANDIDATES:
        if not candidate.exists():
            continue
        try:
            result = subprocess.run(
                ["sops", "--decrypt", "--extract", f'["{key}"]', str(candidate)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # sops not installed or timed out — fall through to env var check
            break

    # Env var fallback: acceptable for local dev and CI/CD environments
    return (
        os.environ.get(key)
        or os.environ.get("HUGGING_FACE_HUB_TOKEN" if key == "HF_TOKEN" else key)
        or None
    )
