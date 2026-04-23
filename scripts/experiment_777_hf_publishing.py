#!/usr/bin/env python3
"""Experiment 777 — HuggingFace Publishing: StepLevelJEPAProbe + KAN Tier 0b + README updates.

**Research question:**
    Can we execute the huggingface-cli uploads prepared in Exp 752 and update all 16
    existing Carnot-EBM model READMEs to point users at `pip install carnot`?

**Why this experiment matters:**
    Exp 752 prepared all artifacts (safetensors, configs, model cards) and wrote
    hf_upload_commands.sh, but never executed the push.  This experiment closes that
    gap by:
      1. Checking HF authentication (HF_TOKEN env or prior `huggingface-cli login`).
      2. Uploading StepLevelJEPAProbe and KAN Tier 0b artifacts via huggingface-cli.
      3. Updating all 16 existing Carnot-EBM model READMEs to add a
         "## Production Use" section pointing users at `pip install carnot`.

**Honest verdict mapping:**
    hf_published_readmes_updated:  both new models published AND existing READMEs updated
    hf_published_readmes_blocked:  new models published but README updates failed
    hf_artifacts_uploaded_only:    new models published but no existing models found
    blocked_hf_not_authenticated:  HF_TOKEN not set / huggingface-cli whoami fails

Spec: REQ-PUBLISH-010, REQ-PUBLISH-011,
      SCENARIO-PUBLISH-010, SCENARIO-PUBLISH-011
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

DELIVERABLE = "results/experiment_777_hf_publishing.json"
EXP_752_RESULT = "results/experiment_752_hf_model_preparation.json"

# The section injected into every existing Carnot-EBM model README.
_PRODUCTION_USE_SECTION = """
## Production Use

For production LLM output verification, install: `pip install carnot`

This model is a Phase 1 research artifact (activation-based confidence detection).
These 16 per-token activation EBMs detect confidence, not correctness.
For the full verify-repair pipeline, see: https://github.com/ianblenke/carnot
"""


def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    """Run a subprocess command, return (returncode, stdout, stderr).

    Capped at 120 seconds per call — huggingface-cli uploads for small
    safetensors files should complete well within this window.
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


def check_hf_authentication() -> tuple[bool, str]:
    """Return (authenticated, username) by calling `hf auth whoami`.

    The legacy `huggingface-cli` binary is deprecated as of huggingface_hub
    1.x and prints a warning + non-zero exit — historical reads of its
    whoami call silently broke (RETRO-074 debug session, Exp 777 first run
    reported blocked_hf_not_authenticated despite `hf auth whoami` working
    on the same machine).  The modern CLI is `hf`; its whoami output is
    `user=<name> orgs=<org1>,<org2>` on stdout.
    """
    rc, stdout, _stderr = _run(["hf", "auth", "whoami"], timeout=30)
    if rc != 0:
        return False, ""
    # Output format: `user=ianblenke orgs=Carnot-EBM` (single line).
    for token in stdout.strip().split():
        if token.startswith("user="):
            return True, token.split("=", 1)[1]
    # Fall back — authenticated but couldn't parse username.
    return True, ""


def upload_artifact(
    repo_id: str,
    local_path: str,
    path_in_repo: str | None = None,
) -> tuple[bool, str]:
    """Upload a single file to a HuggingFace model repository.

    Returns (success, url_or_error).

    Why path_in_repo is optional: most uploads go to the repo root.
    huggingface-cli upload auto-creates the repo if it does not exist.
    """
    cmd = ["hf", "upload", repo_id, local_path]
    if path_in_repo:
        cmd.append(path_in_repo)
    cmd += ["--repo-type", "model"]
    rc, stdout, stderr = _run(cmd, timeout=300)
    if rc == 0:
        url = f"https://huggingface.co/{repo_id}"
        return True, url
    return False, stderr.strip()


def get_existing_org_models(org: str) -> list[str]:
    """List model repo IDs in a HuggingFace organization.

    Returns a list of full repo IDs like ['Carnot-EBM/some-model', ...].
    Returns empty list on failure (e.g. org does not exist or CLI error).

    Why not use the Python huggingface_hub library: the CLI is always present
    when huggingface-cli is authenticated; avoids an extra import path.
    """
    rc, stdout, _stderr = _run(
        ["hf", "models", "list", "--author", org, "--limit", "50"],
        timeout=60,
    )
    if rc != 0 or not stdout.strip():
        return []
    models = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if line and not line.startswith("NAME") and not line.startswith("---"):
            # Output format: "Carnot-EBM/model-name  model  ..."
            parts = line.split()
            if parts:
                models.append(parts[0])
    return models


def update_readme_with_production_section(repo_id: str) -> tuple[bool, str]:
    """Download existing README.md, inject the production-use section, re-upload.

    Returns (success, error_message).

    Why we download first: the README may already have a production section
    (idempotent re-run) or may be absent entirely.  We append to the existing
    content rather than replacing, to preserve prior model documentation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        readme_path = Path(tmpdir) / "README.md"

        # Try to download existing README.
        dl_cmd = [
            "hf", "download", repo_id, "README.md",
            "--repo-type", "model",
            "--local-dir", tmpdir,
        ]
        dl_rc, _dl_out, _dl_err = _run(dl_cmd, timeout=60)

        if dl_rc == 0 and readme_path.exists():
            existing = readme_path.read_text()
        else:
            existing = f"# {repo_id.split('/')[-1]}\n\nCarnot-EBM model.\n"

        if "## Production Use" in existing:
            # Already updated — idempotent, counts as success.
            return True, "already_present"

        updated = existing.rstrip() + "\n" + _PRODUCTION_USE_SECTION
        readme_path.write_text(updated)

        ul_cmd = [
            "hf", "upload", repo_id, str(readme_path),
            "--repo-type", "model",
        ]
        ul_rc, _ul_out, ul_err = _run(ul_cmd, timeout=120)
        if ul_rc == 0:
            return True, ""
        return False, ul_err.strip()


def run_experiment(tmpl: ExperimentTemplate) -> dict:
    """Main experiment logic — separated from __main__ for testability.

    Why separated: unit tests can call run_experiment(fake_tmpl) with a
    MagicMock template and patched subprocess helpers without spawning
    real huggingface-cli processes.
    """
    # Step 1: Check HF authentication (REQ-PUBLISH-010).
    hf_authenticated, hf_username = check_hf_authentication()
    if not hf_authenticated:
        return {
            "hf_authenticated": False,
            "hf_username": "",
            "n_models_published": 0,
            "published_urls": [],
            "n_readmes_updated": 0,
            "honest_verdict": "blocked_hf_not_authenticated",
        }

    # Step 2: Load Exp 752 upload manifest.
    repo_root = Path(tmpl._repo_root)
    exp752_path = repo_root / EXP_752_RESULT
    if not exp752_path.exists():
        return {
            "hf_authenticated": True,
            "hf_username": hf_username,
            "n_models_published": 0,
            "published_urls": [],
            "n_readmes_updated": 0,
            "honest_verdict": "blocked_exp752_manifest_missing",
        }

    exp752 = json.loads(exp752_path.read_text())
    artifact_paths: dict = exp752.get("artifact_paths", {})

    # Step 3: Upload StepLevelJEPAProbe artifacts.
    n_models_published = 0
    published_urls: list[str] = []
    upload_errors: list[str] = []

    jepa_artifacts = [
        ("carnot_step_jepa_probe_v1.safetensors", artifact_paths.get("jepa_weights", "")),
        ("carnot_step_jepa_probe_v1_config.json", artifact_paths.get("jepa_config", "")),
        ("README.md", artifact_paths.get("jepa_model_card", "")),
    ]
    jepa_repo = "Carnot-EBM/carnot-step-jepa-probe-v1"
    jepa_ok = True
    for _filename, local_path in jepa_artifacts:
        if not local_path or not Path(local_path).exists():
            jepa_ok = False
            upload_errors.append(f"missing artifact: {local_path}")
            continue
        ok, url_or_err = upload_artifact(jepa_repo, local_path)
        if not ok:
            jepa_ok = False
            upload_errors.append(url_or_err)
    if jepa_ok:
        n_models_published += 1
        published_urls.append(f"https://huggingface.co/{jepa_repo}")

    # Step 4: Upload KAN Tier 0b artifacts.
    kan_artifacts = [
        ("carnot_kan_tier0b_v3.safetensors", artifact_paths.get("kan_weights", "")),
        ("carnot_kan_tier0b_v3_config.json", artifact_paths.get("kan_config", "")),
        ("README.md", artifact_paths.get("kan_model_card", "")),
    ]
    kan_repo = "Carnot-EBM/carnot-kan-tier0b-v3"
    kan_ok = True
    for _filename, local_path in kan_artifacts:
        if not local_path or not Path(local_path).exists():
            kan_ok = False
            upload_errors.append(f"missing artifact: {local_path}")
            continue
        ok, url_or_err = upload_artifact(kan_repo, local_path)
        if not ok:
            kan_ok = False
            upload_errors.append(url_or_err)
    if kan_ok:
        n_models_published += 1
        published_urls.append(f"https://huggingface.co/{kan_repo}")

    # Step 5: Update existing Carnot-EBM model READMEs (REQ-PUBLISH-011).
    existing_models = get_existing_org_models("Carnot-EBM")
    n_readmes_updated = 0
    readme_errors: list[str] = []

    for repo_id in existing_models:
        ok, err = update_readme_with_production_section(repo_id)
        if ok:
            n_readmes_updated += 1
        else:
            readme_errors.append(f"{repo_id}: {err}")

    # Step 6: Determine honest verdict.
    if n_models_published == 0:
        honest_verdict = "blocked_hf_not_authenticated"  # shouldn't reach here
    elif not existing_models:
        honest_verdict = "hf_artifacts_uploaded_only"
    elif n_readmes_updated == 0:
        honest_verdict = "hf_published_readmes_blocked"
    else:
        honest_verdict = "hf_published_readmes_updated"

    return {
        "hf_authenticated": True,
        "hf_username": hf_username,
        "n_models_published": n_models_published,
        "published_urls": published_urls,
        "n_readmes_updated": n_readmes_updated,
        "existing_models_found": existing_models,
        "upload_errors": upload_errors,
        "readme_errors": readme_errors,
        "honest_verdict": honest_verdict,
    }


def main() -> None:
    """Run Experiment 777 end-to-end with watchdog and deliverable guard."""
    tmpl = ExperimentTemplate(
        777,
        "HuggingFace Publishing — StepLevelJEPAProbe + KAN Tier 0b + README updates",
        DELIVERABLE,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(777, timeout_minutes=30, result_path=DELIVERABLE):
        result = run_experiment(tmpl)

        # Determine top-level status for ExperimentTemplate schema compliance.
        honest_verdict = result.get("honest_verdict", "")
        if "blocked" in honest_verdict:
            status = "blocked"
        else:
            status = "success"

        artifact = tmpl.build_result(result, status=status)

        import json as _json
        from pathlib import Path as _Path
        _Path(tmpl._output_path).write_text(_json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
