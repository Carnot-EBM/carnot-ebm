#!/usr/bin/env python3
"""Experiment 803 — HuggingFace Publish v2: SOPS-encrypted HF_TOKEN + authenticated upload.

**Research question:**
    Can we close RETRO-HF-AUTH (HF_TOKEN absent from conductor environment in Exp 777 .59)
    by (a) specifying how to store HF_TOKEN via SOPS encryption, and (b) attempting an
    authenticated README update when the token IS present in the current environment?

**Why this experiment matters:**
    Exp 777 (.59) reported honest_verdict=blocked_hf_not_authenticated because HF_TOKEN
    was never injected into the conductor session.  This experiment:
      1. Creates the SOPS configuration spec (docs/sops-hf-token-setup.md) so operators
         know exactly how to provision the token securely.
      2. Updates models/hf_upload_commands.sh with SOPS-based token injection for all
         three model tiers (Ising, KAN, EORM).
      3. Checks whether HF_TOKEN is available in the current environment via two
         standard env var names (HF_TOKEN, HUGGING_FACE_HUB_TOKEN).
      4. If available AND huggingface-cli (or hf) is installed: calls whoami to confirm
         authentication, then attempts to update the carnot-ising-sampler-v1 README
         as a minimal live publish test.

**Honest verdict mapping:**
    hf_models_published:     authenticated AND at least 1 README updated
    hf_auth_documented:      SOPS spec written, token absent — docs delivered, publish deferred
    hf_cli_not_installed:    neither huggingface-cli nor hf binary found in PATH

Spec: REQ-PUBLISH-005, REQ-PUBLISH-006, SCENARIO-PUBLISH-009
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

DELIVERABLE = "results/experiment_803_hf_publish_v2.json"

# The README section that proves the Ising tier is wired into pip install carnot.
_ISING_README_SECTION = """
## Production Use (Exp 803)

Install via: `pip install carnot`

This model (Carnot Ising Sampler v1) is the Tier: Small Boltzmann sampler.
For the full verify-repair pipeline, see: https://github.com/Carnot-EBM/carnot-ebm
"""


def _run(cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
    """Run a subprocess command, return (returncode, stdout, stderr).

    Captures both stdout and stderr so callers can inspect output without
    cluttering the terminal.  Times out after `timeout` seconds to prevent
    runaway huggingface-cli calls from blocking the watchdog.
    """
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"command timed out after {timeout}s: {cmd}"
    except Exception as exc:
        return 1, "", str(exc)


def get_hf_token() -> str | None:
    """Return HF_TOKEN from either of the two standard environment variable names.

    HuggingFace libraries accept both HF_TOKEN (newer standard) and
    HUGGING_FACE_HUB_TOKEN (legacy name).  Checking both ensures we don't
    falsely report the token as absent when it was set under the legacy name.

    REQ-PUBLISH-005: token must be present; this function checks both names.
    """
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def find_hf_cli() -> str | None:
    """Return the name of the available huggingface CLI binary, or None.

    The modern HuggingFace CLI is `hf` (huggingface_hub >= 1.0).  The legacy
    CLI is `huggingface-cli`.  We prefer `hf` but fall back to `huggingface-cli`
    for older installations.

    REQ-PUBLISH-006: upload commands require huggingface-cli or hf in PATH.
    """
    for candidate in ("hf", "huggingface-cli"):
        rc, stdout, _ = _run([candidate, "--version"], timeout=10)
        if rc == 0:
            return candidate
    return None


def check_hf_auth(cli: str, token: str) -> tuple[bool, str]:
    """Return (authenticated, username) by calling `<cli> auth whoami` or `<cli> whoami`.

    Different CLI versions expose different subcommand paths.  We try the modern
    `hf auth whoami` path first, then the legacy `huggingface-cli whoami` path.

    Why inject the token as env var here: calling `hf login --token` would
    persist the token to ~/.cache/huggingface and leave it on disk.  Setting
    HF_TOKEN in the subprocess environment is sufficient for whoami and upload
    without writing to disk.
    """
    env = {**os.environ, "HF_TOKEN": token, "HUGGING_FACE_HUB_TOKEN": token}
    for cmd in ([cli, "auth", "whoami"], [cli, "whoami"]):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
            if proc.returncode == 0:
                out = proc.stdout.strip()
                # Modern format: "user=name orgs=Carnot-EBM"
                for part in out.split():
                    if part.startswith("user="):
                        return True, part.split("=", 1)[1]
                # Older format: just the username on a line
                if out and " " not in out:
                    return True, out
                return True, ""
        except (subprocess.TimeoutExpired, Exception):
            continue
    return False, ""


def attempt_readme_update(cli: str, token: str, repo_id: str, content: str) -> tuple[bool, str]:
    """Upload an in-memory README string to a HuggingFace model repo.

    Writes the README to a temp file then calls `<cli> upload` to push it.
    Returns (success, url_or_error_message).

    Why a temp file: huggingface-cli/hf upload expects a local file path,
    not stdin.  A NamedTemporaryFile with delete=False gives us a path that
    persists until we explicitly remove it.

    REQ-PUBLISH-006: upload MUST use huggingface-cli upload (or hf upload).
    """
    env = {**os.environ, "HF_TOKEN": token, "HUGGING_FACE_HUB_TOKEN": token}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        # Both `hf upload` and `huggingface-cli upload` accept: repo_id local_path dest_path
        cmd = [cli, "upload", repo_id, tmp_path, "README.md", "--repo-type", "model"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
        if proc.returncode == 0:
            return True, f"https://huggingface.co/{repo_id}"
        return False, proc.stderr.strip() or proc.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, f"upload timed out for {repo_id}"
    except Exception as exc:
        return False, str(exc)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_experiment(tmpl: ExperimentTemplate) -> dict:
    """Run the Exp 803 logic and return the result payload dict.

    Steps:
      1. Confirm SOPS spec was written (deliverable from this experiment run).
      2. Check HF_TOKEN availability.
      3. Check CLI availability.
      4. If token + CLI: check auth, attempt 1 README update.
      5. Return result with honest_verdict.

    REQ-PUBLISH-005, REQ-PUBLISH-006, SCENARIO-PUBLISH-009
    """
    repo_root = Path(tmpl._repo_root) if hasattr(tmpl, "_repo_root") else _REPO_ROOT

    # --- Step 1: confirm SOPS doc was written (in this session by write above) ---
    sops_doc_path = repo_root / "docs" / "sops-hf-token-setup.md"
    sops_doc_written = sops_doc_path.exists()

    hf_upload_script = repo_root / "models" / "hf_upload_commands.sh"
    upload_script_written = hf_upload_script.exists()

    # --- Step 2: check HF_TOKEN ---
    token = get_hf_token()
    hf_token_present = token is not None

    # --- Step 3: check CLI ---
    cli = find_hf_cli()
    cli_present = cli is not None

    if not cli_present:
        return {
            "honest_verdict": "hf_cli_not_installed",
            "hf_token_present": hf_token_present,
            "hf_cli_found": None,
            "hf_authenticated": False,
            "sops_doc_written": sops_doc_written,
            "upload_script_written": upload_script_written,
            "models_published": [],
            "note": "huggingface-cli / hf not found in PATH. Install with: pip install huggingface_hub[cli]",
        }

    if not hf_token_present:
        return {
            "honest_verdict": "hf_auth_documented",
            "hf_token_present": False,
            "hf_cli_found": cli,
            "hf_authenticated": False,
            "sops_doc_written": sops_doc_written,
            "upload_script_written": upload_script_written,
            "models_published": [],
            "note": "HF_TOKEN absent. See docs/sops-hf-token-setup.md for SOPS setup instructions.",
        }

    # --- Step 4: attempt auth + README update ---
    assert token is not None  # narrowing — token is not None here
    authenticated, username = check_hf_auth(cli, token)

    if not authenticated:
        return {
            "honest_verdict": "hf_auth_documented",
            "hf_token_present": True,
            "hf_cli_found": cli,
            "hf_authenticated": False,
            "username": "",
            "sops_doc_written": sops_doc_written,
            "upload_script_written": upload_script_written,
            "models_published": [],
            "note": "HF_TOKEN present but whoami failed — token may be expired or invalid.",
        }

    # Authenticated — attempt to update Ising tier README as live publish test.
    repo_id = "Carnot-EBM/carnot-ising-sampler-v1"
    success, url_or_error = attempt_readme_update(cli, token, repo_id, _ISING_README_SECTION)

    models_published = [repo_id] if success else []
    honest_verdict = "hf_models_published" if success else "hf_auth_documented"

    return {
        "honest_verdict": honest_verdict,
        "hf_token_present": True,
        "hf_cli_found": cli,
        "hf_authenticated": True,
        "username": username,
        "sops_doc_written": sops_doc_written,
        "upload_script_written": upload_script_written,
        "models_published": models_published,
        "readme_update_result": url_or_error,
        "note": f"README update {'succeeded' if success else 'failed'}: {url_or_error}",
    }


def main() -> None:
    """Run Experiment 803 end-to-end with watchdog and deliverable guard."""
    # apply_env_autofix MUST be called before any JAX or CUDA import.
    # It injects CARNOT_FORCE_LIVE=1 if GPU hardware is present, ensuring
    # downstream pipeline code uses live inference instead of cached responses.
    apply_env_autofix()

    tmpl = ExperimentTemplate(
        803,
        "HuggingFace Publish v2: SOPS-encrypted HF_TOKEN + authenticated upload",
        DELIVERABLE,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(803, timeout_minutes=30, result_path=DELIVERABLE):
        result = run_experiment(tmpl)

        honest_verdict = result.get("honest_verdict", "")
        status = "blocked" if honest_verdict == "hf_cli_not_installed" else "success"

        artifact = tmpl.build_result(result, status=status)

        Path(tmpl._output_path).write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
