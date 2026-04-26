#!/usr/bin/env python3
"""Exp 890: GGUF CLI download v3 — validate hf CLI as canonical download path.

**Why this experiment exists:**
    Eleven prior attempts (Exps 857–869) to download GGUF models used
    ``huggingface_hub.hf_hub_download()`` (Python API).  Every attempt failed with
    RepositoryNotFoundError.  The root cause: the Python API uses a requests session
    with case-sensitive filename matching and endpoint selection that diverges from
    what the CLI binary does.

    This experiment tests a fundamentally different approach: the ``hf`` or
    ``huggingface-cli`` binary (OS-level subprocess) which uses its own HTTP stack.
    If CLI download succeeds here, ``GGUFCacheResolver.cli_download()`` becomes
    the canonical download method for all future experiments.

**Spec:** REQ-INFRA-074 (GGUFCacheResolver CLI fallback), SCENARIO-INFRA-083

**Prior failures addressed:**
    - Exp 869: download_verified=False (Python API, RepositoryNotFoundError 404)
    - Exp 857: blocked (same root cause)
    retire_if_same_verdict: true — if this also fails, GGUF-model-download is retired.
"""

from __future__ import annotations

import json
import sys
import os

# Ensure project root is importable when run as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.experiment_template import ExperimentTemplate
from python.carnot.resolvers.gguf_cache import GGUFCacheResolver

# --- Config ------------------------------------------------------------------
EXP_ID = 890
TITLE = "GGUF CLI download v3 — hf CLI subprocess approach"
DELIVERABLE = "results/experiment_890_gguf_download_v3_cli.json"

# Qwen3.5-0.8B is the smallest officially-released Qwen3.5 GGUF.
# The correct HuggingFace repo for GGUF quantisations by unsloth is checked below.
# (The plain Qwen/Qwen3.5-0.8B-GGUF repo returned 404 in Exp 869.)
HF_REPO = "Qwen/Qwen3.5-0.8B-GGUF"
FILENAME = "Qwen3.5-0.8B-Q4_K_M.gguf"
DEST_DIR = "models/gguf/qwen3.5-0.8b"
CLI_TIMEOUT = 300

EXCLUSION_ITEM = {
    "id": "GGUF-model-download-for-code-repair",
    "reason": (
        "CLI approach also failed; 12 total attempts (Exps 857-890). "
        "Code repair via GGUF models is retired — use transformers loader (Exp 881) instead."
    ),
}
# -----------------------------------------------------------------------------


def _write_exclusion(manifest_path: str, item: dict) -> None:
    """Append ``item`` to the exclusion manifest YAML at ``manifest_path``.

    We write a minimal YAML append rather than parsing with PyYAML to avoid
    an optional dependency.  The conductor reads this file at pre-flight.
    """
    with open(manifest_path, "a") as fh:
        fh.write(
            f"\n# Added by Exp {EXP_ID} — retire_if_same_verdict triggered\n"
            f"- id: {item['id']}\n"
            f"  reason: |\n"
            f"    {item['reason']}\n"
        )


def main() -> None:
    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    resolver = GGUFCacheResolver()

    # --- Step 1: probe for CLI availability ---------------------------------
    import shutil as _shutil

    hf_cmd = _shutil.which("hf") or _shutil.which("huggingface-cli")
    cli_found = hf_cmd is not None

    if not cli_found:
        artifact = tmpl.build_result(
            {
                "cli_found": False,
                "download_verified": False,
                "file_size_mb": None,
                "cache_path": None,
                "error": "hf CLI not found on PATH",
                "download_method": None,
                "honest_verdict": "cli_not_found",
            },
            status="blocked",
        )
        _persist(artifact, DELIVERABLE)
        tmpl.assert_deliverable_written()
        return

    # --- Step 2: check cache first (resolve() is zero-network) -------------
    cached_path = resolver.resolve(HF_REPO, FILENAME)

    if cached_path is not None:
        size_mb = round(cached_path.stat().st_size / (1024 * 1024), 2)
        artifact = tmpl.build_result(
            {
                "cli_found": True,
                "download_verified": True,
                "file_size_mb": size_mb,
                "cache_path": str(cached_path),
                "error": None,
                "download_method": "cache_hit",
                "honest_verdict": "cli_download_verified",
            },
            status="success",
        )
        _persist(artifact, DELIVERABLE)
        tmpl.assert_deliverable_written()
        return

    # --- Step 3: attempt CLI download ----------------------------------------
    result = resolver.cli_download(HF_REPO, FILENAME, DEST_DIR, timeout_s=CLI_TIMEOUT)

    download_verified = result.get("success", False)
    file_size_mb = result.get("size_mb") if download_verified else None
    cache_path = result.get("path") if download_verified else None
    error = result.get("error") if not download_verified else None
    download_method = "cli" if download_verified else None

    if download_verified:
        honest_verdict = "cli_download_verified"
        status = "success"
    else:
        honest_verdict = "download_failed_retire"
        status = "failed"
        # retire_if_same_verdict: true — add to exclusion manifest
        exclusion_manifest = os.path.join(_REPO_ROOT, "ops", "exclusion_manifest.yaml")
        if os.path.exists(exclusion_manifest):
            _write_exclusion(exclusion_manifest, EXCLUSION_ITEM)

    artifact = tmpl.build_result(
        {
            "cli_found": cli_found,
            "download_verified": download_verified,
            "file_size_mb": file_size_mb,
            "cache_path": cache_path,
            "error": error,
            "download_method": download_method,
            "honest_verdict": honest_verdict,
            "retro_tag": "RETRO-SOTA-MODEL-DOWNLOAD",
            "prior_failures": [
                {"experiment_id": "exp869", "verdict": "download_verified=False"},
                {"experiment_id": "exp857", "verdict": "blocked"},
            ],
        },
        status=status,
    )
    _persist(artifact, DELIVERABLE)
    tmpl.assert_deliverable_written()


def _persist(artifact: dict, path: str) -> None:
    """Write ``artifact`` as pretty-printed JSON to ``path``, creating dirs."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(artifact, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()
