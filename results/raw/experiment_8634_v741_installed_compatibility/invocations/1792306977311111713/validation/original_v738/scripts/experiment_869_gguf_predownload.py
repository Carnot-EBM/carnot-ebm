#!/usr/bin/env python3
"""Exp 869 — GGUF pre-download verification.

**Researcher summary:**
    Tests GGUFCacheResolver.pre_download_and_verify() on a small model
    (Qwen/Qwen3.5-0.8B-GGUF, ~500MB) to prove the download mechanism works
    end-to-end before trusting it for 20GB+ SOTA models in Exp 870.

    RETRO-SOTA-MODEL-DOWNLOAD: Exp 857 called download() at runtime and it
    failed silently.  The root cause was unknown (huggingface_hub timeout?
    missing auth token? wrong filename?).  This experiment makes failure
    explicit and diagnosable so Exp 870 can gate on download_verified=True.

**What this experiment does:**
    1. Imports huggingface_hub and verifies it is installed.
    2. Calls pre_download_and_verify() on a known-small GGUF
       (qwen3.5-0.8b-q4_k_m.gguf from Qwen/Qwen3.5-0.8B-GGUF).
    3. Reports success/failure with the exact error if it fails.
    4. Sets download_verified=True in the artifact if the file is on disk
       and > 0 bytes after the call.

Spec: REQ-INFRA-073, SCENARIO-INFRA-082
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow running from project root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Also allow scripts/ dir imports (ExperimentTemplate).
sys.path.insert(0, os.path.dirname(__file__))

from experiment_template import ExperimentTemplate  # noqa: E402

from carnot.pipeline.gguf_cache import GGUFCacheConfig, GGUFCacheResolver  # noqa: E402

# ── Experiment identity ────────────────────────────────────────────────────────
EXP_ID = 869
TITLE = "GGUF pre-download verification (small model smoke-test)"
DELIVERABLE = "results/experiment_869_gguf_predownload.json"

# ── Download target ────────────────────────────────────────────────────────────
# Qwen3.5-0.8B is the smallest model in routine use (~500MB Q4_K_M).
# Using it here instead of a 20GB SOTA model so the test completes quickly
# and proves the mechanism without burning disk quota.
HF_REPO = "Qwen/Qwen3.5-0.8B-GGUF"
# Actual filename inside the HF repo — confirmed from repo file listing.
FILENAME = "qwen3.5-0.8b-q4_k_m.gguf"
DEST_DIR = "models/"


def main() -> None:
    """Run Exp 869: verify GGUFCacheResolver.pre_download_and_verify() end-to-end."""
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    # ── Check huggingface_hub availability ────────────────────────────────────
    hf_hub_available = False
    try:
        import huggingface_hub  # noqa: F401

        hf_hub_available = True
    except ImportError:
        pass

    if not hf_hub_available:
        artifact = tmpl.build_result(
            {
                "download_verified": False,
                "model_id": HF_REPO,
                "filename": FILENAME,
                "file_size_mb": None,
                "cache_path": None,
                "hf_hub_available": False,
                "error": "huggingface_hub is not installed — cannot download",
                "honest_verdict": "hf_hub_missing",
                "retro_tag": "RETRO-SOTA-MODEL-DOWNLOAD",
                "gates_exp_870": False,
            },
            status="blocked",
        )
        Path(DELIVERABLE).write_text(json.dumps(artifact, indent=2))
        print(f"[Exp {EXP_ID}] BLOCKED: huggingface_hub not installed")
        tmpl.assert_deliverable_written()
        return

    # ── Run pre_download_and_verify() ─────────────────────────────────────────
    config = GGUFCacheConfig(cache_dir=DEST_DIR)
    resolver = GGUFCacheResolver(config)

    print(
        f"[Exp {EXP_ID}] Calling pre_download_and_verify({HF_REPO!r}, {FILENAME!r}, {DEST_DIR!r})"
    )
    result = resolver.pre_download_and_verify(HF_REPO, FILENAME, DEST_DIR)

    download_verified = result["success"]
    honest_verdict = "download_verified" if download_verified else "download_failed"

    artifact = tmpl.build_result(
        {
            "download_verified": download_verified,
            "model_id": HF_REPO,
            "filename": FILENAME,
            "file_size_mb": result.get("size_mb"),
            "cache_path": result.get("path"),
            "hf_hub_available": hf_hub_available,
            "error": result.get("error"),
            "honest_verdict": honest_verdict,
            "retro_tag": "RETRO-SOTA-MODEL-DOWNLOAD",
            # download_verified=True means Exp 870 can safely rely on this path.
            "gates_exp_870": download_verified,
        },
        status="success" if download_verified else "failed",
    )
    Path(DELIVERABLE).write_text(json.dumps(artifact, indent=2))

    if download_verified:
        print(
            f"[Exp {EXP_ID}] SUCCESS — download_verified=True, "
            f"size={result['size_mb']:.1f} MB, path={result['path']}"
        )
    else:
        print(f"[Exp {EXP_ID}] FAILED — download_verified=False, error={result['error']!r}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
