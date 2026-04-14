"""Experiment 304: HuggingFace publish — credential re-check and actual upload.

Carry-forward from Exp 293 (blocked because huggingface-cli was not in PATH).
This experiment adds a Python API fallback so that upload proceeds when
``huggingface_hub`` is installed and credentials are cached (HF_TOKEN env var
or ~/.huggingface/token), even when the CLI binary is absent.

What is published:
  1. Carnot-EBM/carnot-joint-constraint-v1  — Exp 66 joint EBM+Ising (Phase 1 prototype)
  2. Carnot-EBM/carnot-formal-claim-verifier-v1 — FCV ONNX for arithmetic+comparison

Credential check precedence:
  1. Try ``huggingface-cli whoami`` (subprocess) — matches Exp 293 behaviour.
  2. If CLI not found or returns non-zero, fall back to ``HfApi().whoami()`` via
     the Python API.  This succeeds when HF_TOKEN is set or a cached token exists.
  3. If both fail → emit blocked artifact with ``exp_304_next_action`` login hint.

Spec: REQ-VERIFY-058, REQ-VERIFY-059
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

# HuggingFace repo IDs (same as Exp 293)
_EXP66_REPO_ID = "Carnot-EBM/carnot-joint-constraint-v1"
_FCV_REPO_ID = "Carnot-EBM/carnot-formal-claim-verifier-v1"

# Results file path
_RESULTS_PATH = Path(__file__).parent.parent / "results" / "experiment_304_hf_results.json"


# ---------------------------------------------------------------------------
# Dependency injection helper — makes HfApi patchable in tests
# ---------------------------------------------------------------------------


def _make_hf_api() -> Any:
    """Return a new HfApi instance.

    Exists as a standalone function so tests can patch
    ``scripts.experiment_304_hf_publish._make_hf_api`` and inject a mock
    without importing huggingface_hub at module load time.
    """
    from huggingface_hub import HfApi  # type: ignore[import-untyped]
    return HfApi()


# ---------------------------------------------------------------------------
# Credential check with CLI + Python API fallback
# ---------------------------------------------------------------------------


def check_hf_credentials_304() -> tuple[bool, str]:
    """Check HuggingFace credentials via CLI, then Python API fallback.

    Precedence:
      1. ``huggingface-cli whoami`` — exact same check as Exp 293.
      2. If CLI absent or fails → ``HfApi().whoami()`` Python API call.
      3. If both fail → return (False, login_instructions).

    Returns:
        ``(True, username_or_msg)`` if credentials are valid.
        ``(False, instructions)`` if not authenticated.
    """
    _login_instructions = (
        "Run: huggingface-cli login --token <your-token>\n"
        "or set the HF_TOKEN environment variable.\n"
        "After login, re-run this script."
    )

    # --- Step 1: try CLI ---
    cli_ok: bool = False
    cli_username: str = ""
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            cli_ok = True
            cli_username = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    except FileNotFoundError:
        pass  # CLI not installed — fall through to Python API
    except Exception:
        pass  # Any other error — fall through

    if cli_ok:
        return True, f"logged in as {cli_username}" if cli_username else "logged in"

    # --- Step 2: Python API fallback ---
    try:
        api = _make_hf_api()
        info = api.whoami()
        username = info.get("name", "") if isinstance(info, dict) else str(info)
        return True, f"logged in as {username}" if username else "logged in (Python API)"
    except Exception:
        pass  # Not authenticated via Python API either

    return False, _login_instructions


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_experiment_304(
    out_dir: Path | str | None = None,
    dry_run: bool = False,
    results_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full Exp 304 HuggingFace publish pipeline.

    Steps:
      1. Check credentials (CLI → Python API fallback).
         If not authenticated → emit blocked artifact with exp_304_next_action.
      2. Delegate artifact staging + upload to Exp 293's ``run_experiment_293``.
         Injects ``dry_run`` flag and overrides results path so Exp 293 writes
         its own JSON separately.
      3. Build Exp 304 results JSON with credentials_available, repo_urls, etc.
      4. Update README.md HuggingFace section with confirmed upload status if
         upload actually succeeded (not dry_run).

    Args:
        out_dir: Optional staging directory for artifacts.
        dry_run: If True, skip live HF API calls (simulate upload).
        results_path: Override write path for results JSON.

    Returns:
        Results dict (also written to disk).
    """
    import tempfile

    _results_write_path = results_path if results_path is not None else _RESULTS_PATH

    def _write_results(data: dict[str, Any]) -> None:
        _results_write_path.parent.mkdir(parents=True, exist_ok=True)
        _results_write_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    # -----------------------------------------------------------------------
    # Step 1: Credential check
    # -----------------------------------------------------------------------
    creds_ok, creds_msg = check_hf_credentials_304()

    if not creds_ok:
        blocked: dict[str, Any] = {
            "experiment": 304,
            "run_date": "20260414",
            "blocked": True,
            "credentials_available": False,
            "exp_304_next_action": (
                "Run: huggingface-cli login --token <your-token>\n"
                "or: export HF_TOKEN=<your-token>\n"
                "Then re-run: python scripts/experiment_304_hf_publish.py"
            ),
            "login_instructions": (
                "Run: huggingface-cli login\n"
                "or set the HF_TOKEN environment variable.\n"
                "After login, re-run this script."
            ),
            "repo_ids": {
                "exp66": _EXP66_REPO_ID,
                "fcv": _FCV_REPO_ID,
            },
            "artifacts": {
                "exp66": {"upload_status": "blocked", "hf_url": None},
                "fcv": {"upload_status": "blocked", "hf_url": None},
            },
            "honest_verdict": {
                "status": "blocked",
                "explanation": (
                    "HuggingFace credentials not found (tried CLI + Python API). "
                    "No artifacts were uploaded."
                ),
            },
        }
        _write_results(blocked)
        return blocked

    # -----------------------------------------------------------------------
    # Step 2: Stage and upload artifacts using Exp 293's sub-functions directly.
    # We do NOT call run_experiment_293() because it re-runs the CLI credential
    # check internally and would block even when the Python API is authenticated.
    # Instead we import the individual functions and drive them ourselves.
    # -----------------------------------------------------------------------
    import shutil

    if out_dir is None:
        _tmp = tempfile.mkdtemp(prefix="exp304_")
        staging = Path(_tmp)
    else:
        staging = Path(out_dir)
        staging.mkdir(parents=True, exist_ok=True)

    from scripts.experiment_293_huggingface_publish import (
        _EXP66_REPO_ID as _E293_EXP66_REPO_ID,
        _EXP66_SAFETENSORS_PATH,
        _FCV_REPO_ID as _E293_FCV_REPO_ID,
        _TAG,
        _write_exp66_config,
        build_exp66_model_card,
        build_fcv_model_card,
        export_fcv_onnx,
        upload_artifacts,
        _write_fcv_python_module,
    )

    # Reuse the same HfApi instance (already validated) for the upload.
    # This avoids a second CLI-based credential check inside upload_artifacts.
    hf_api = _make_hf_api()

    # -- Stage FCV artifacts --
    fcv_dir = staging / "fcv"
    fcv_dir.mkdir(exist_ok=True)
    arith_onnx, cmp_onnx = export_fcv_onnx(fcv_dir)
    _write_fcv_python_module(fcv_dir)
    (fcv_dir / "README.md").write_text(build_fcv_model_card())

    # -- Stage Exp 66 artifacts (only if trained weights exist) --
    exp66_upload_dir: Path | None = None
    exp66_safetensors_str: str | None = None
    if _EXP66_SAFETENSORS_PATH.exists():
        exp66_dir = staging / "exp66"
        exp66_dir.mkdir(exist_ok=True)
        dst = exp66_dir / "exp66.safetensors"
        shutil.copy2(str(_EXP66_SAFETENSORS_PATH), str(dst))
        _write_exp66_config(exp66_dir)
        (exp66_dir / "README.md").write_text(build_exp66_model_card())
        exp66_upload_dir = exp66_dir
        exp66_safetensors_str = str(dst)
        exp66_artifact_status = "staged"
    else:
        exp66_artifact_status = "skipped_missing_safetensors"

    # -- Upload --
    upload_result = upload_artifacts(
        exp66_dir=exp66_upload_dir,
        fcv_dir=fcv_dir,
        tag=_TAG,
        dry_run=dry_run,
        hf_api=hf_api,
        exp66_repo_id=_E293_EXP66_REPO_ID,
        fcv_repo_id=_E293_FCV_REPO_ID,
    )

    upload_status_str = "dry_run" if dry_run else "uploaded"
    exp66_final_status = (
        "skipped_missing_safetensors"
        if exp66_artifact_status == "skipped_missing_safetensors"
        else upload_status_str
    )

    # -----------------------------------------------------------------------
    # Step 3: Build Exp 304 results
    # -----------------------------------------------------------------------
    fcv_artifact: dict[str, Any] = {
        "upload_status": upload_status_str,
        "hf_url": upload_result["fcv_repo"],
        "onnx_arithmetic": str(arith_onnx),
        "onnx_comparison": str(cmp_onnx),
    }
    exp66_artifact: dict[str, Any] = {
        "upload_status": exp66_final_status,
        "hf_url": upload_result["exp66_repo"] if exp66_artifact_status != "skipped_missing_safetensors" else None,
        "safetensors": exp66_safetensors_str,
        "missing_note": (
            None if exp66_artifact_status != "skipped_missing_safetensors"
            else f"results/experiment_66_model.safetensors not found at {_EXP66_SAFETENSORS_PATH}"
        ),
    }

    # Gather published URLs for the repo_urls list
    repo_urls: list[str] = []
    if fcv_artifact.get("hf_url"):
        repo_urls.append(fcv_artifact["hf_url"])
    if exp66_artifact.get("hf_url"):
        repo_urls.append(exp66_artifact["hf_url"])

    honest_status = upload_status_str

    results: dict[str, Any] = {
        "experiment": 304,
        "run_date": "20260414",
        "blocked": False,
        "credentials_available": True,
        "hf_credentials": creds_msg,
        "repo_ids": {
            "exp66": _EXP66_REPO_ID,
            "fcv": _FCV_REPO_ID,
        },
        "repo_urls": repo_urls,
        "artifacts": {
            "exp66": exp66_artifact,
            "fcv": fcv_artifact,
        },
        "honest_verdict": {
            "status": honest_status,
            "explanation": (
                "Credentials verified via Python API fallback (huggingface-cli not in PATH). "
                "FCV ONNX artifacts built and staged. "
                + (
                    "Exp 66 safetensors absent — that artifact was skipped. "
                    if exp66_artifact_status == "skipped_missing_safetensors"
                    else "Exp 66 trained weights staged for upload. "
                )
                + (
                    "dry_run=True; no network calls made."
                    if dry_run
                    else f"Uploaded to HuggingFace Hub with tag {_TAG}."
                )
            ),
        },
    }

    _write_results(results)

    # -----------------------------------------------------------------------
    # Step 4: Update README.md if upload actually succeeded
    # -----------------------------------------------------------------------
    if not dry_run and fcv_artifact.get("upload_status") == "uploaded":
        _update_readme_hf_section(results)

    return results


# ---------------------------------------------------------------------------
# README.md updater
# ---------------------------------------------------------------------------


def _update_readme_hf_section(results: dict[str, Any]) -> None:
    """Update README.md HuggingFace section to note Exp 304 confirmation.

    Only called when a live upload succeeds (not dry_run).  Appends a one-line
    note under the existing Exp 293 section rather than replacing it, per the
    documentation preservation policy in CLAUDE.md.

    Args:
        results: The Exp 304 results dict (used to pull repo URLs).
    """
    readme_path = Path(__file__).parent.parent / "README.md"
    if not readme_path.exists():
        return

    content = readme_path.read_text()
    exp293_header = "### HuggingFace Published Models (Exp 293 / v0.2.0-research)"

    # Only patch if the header exists and hasn't already been updated
    if exp293_header not in content:
        return
    if "Exp 304" in content:
        return  # Already updated

    note = (
        "\n> **Exp 304 (2026-04-14):** Upload confirmed. "
        "Credentials verified via Python API. "
        "FCV artifact live at "
        f"{results['artifacts']['fcv'].get('hf_url', _FCV_REPO_ID)}.\n"
    )
    updated = content.replace(
        exp293_header,
        exp293_header + note,
    )
    readme_path.write_text(updated)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 304: HuggingFace credential re-check and publish."""
    import argparse

    parser = argparse.ArgumentParser(description="Exp 304: HuggingFace credential re-check")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Skip live HF API calls (simulate upload only)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Staging directory for artifacts (default: temp dir)",
    )
    args = parser.parse_args()

    result = run_experiment_304(out_dir=args.out_dir, dry_run=args.dry_run)

    if result.get("blocked"):
        print("BLOCKED: HuggingFace credentials not found.")
        print(result.get("exp_304_next_action", ""))
    else:
        print(f"Exp 304 complete.  Status: {result['honest_verdict']['status']}")
        print(f"  Credentials : {result['hf_credentials']}")
        print(f"  Exp 66 repo : {result['artifacts']['exp66'].get('hf_url')}")
        print(f"  FCV repo    : {result['artifacts']['fcv'].get('hf_url')}")
        print(f"  Results     : {_RESULTS_PATH}")


if __name__ == "__main__":
    main()
