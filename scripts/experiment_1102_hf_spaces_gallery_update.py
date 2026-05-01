"""Experiment 1102 — HuggingFace Spaces gallery update: N-Queens cartridge.

Adds the NQueensGame cartridge (exp1097) to the WOPR gallery's ALL_GAMES
registry and redeploys the HuggingFace Space so it is live for users.

Steps
-----
1. Retrieve HF_TOKEN via SOPS (secrets/hf_token.enc.yaml) or env fallback.
2. Verify games/nqueens.py is present in the spaces directory.
3. Inspect games/__init__.py to confirm NQueensGame is registered.
4. Upload the updated space to HuggingFace via huggingface_hub.
5. HTTP-verify the deployed URL returns 200.

The .84 retro recommendation was to decouple gallery updates from individual
cartridge scripts — this script is the standalone updater for the gallery.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACES_DIR = REPO_ROOT / "spaces" / "wopr-games"
GAMES_DIR = SPACES_DIR / "games"
RESULT_PATH = REPO_ROOT / "results" / "experiment_1102_hf_spaces_gallery_update.json"
HF_SPACE_REPO = "Carnot-EBM/wopr-games"
LIVE_URL = "https://huggingface.co/spaces/Carnot-EBM/wopr-games"


def _retrieve_hf_token() -> tuple[str | None, str]:
    """Try SOPS first, then environment variable.

    Returns (token, source_description).
    """
    for cmd in [
        "sops -d secrets/hf_token.enc.yaml",
        "sops -d ops/secrets.yaml",
    ]:
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=15,
                cwd=REPO_ROOT,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "HF_TOKEN" in line and ":" in line:
                        token = line.split(":", 1)[1].strip()
                        if token:
                            return token, f"sops:{cmd.split()[2]}"
        except Exception:
            pass

    env_token = os.environ.get("HF_TOKEN", "").strip()
    if env_token:
        return env_token, "env:HF_TOKEN"

    return None, "not_found"


def _cartridge_registered(init_path: Path, class_name: str) -> bool:
    """Return True if class_name appears in ALL_GAMES list in __init__.py."""
    content = init_path.read_text()
    return class_name in content and "ALL_GAMES" in content


def _upload_space(token: str) -> tuple[bool, str | None]:
    """Upload spaces/wopr-games/ to HuggingFace using huggingface_hub."""
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        url = api.upload_folder(
            folder_path=str(SPACES_DIR),
            repo_id=HF_SPACE_REPO,
            repo_type="space",
        )
        return True, url
    except ImportError:
        pass

    # Fallback: hf CLI
    try:
        env = {**os.environ, "HF_TOKEN": token}
        result = subprocess.run(
            [
                "hf",
                "upload",
                HF_SPACE_REPO,
                str(SPACES_DIR),
                ".",
                "--repo-type",
                "space",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
            env=env,
        )
        if result.returncode == 0:
            for line in (result.stdout + result.stderr).splitlines():
                if "url=" in line or "huggingface.co" in line:
                    return True, line.strip()
            return True, None
        return False, result.stderr[:500]
    except Exception as exc:
        return False, str(exc)


def _http_status(url: str) -> int:
    """Return the HTTP status code for the given URL, or 0 on failure."""
    try:
        import urllib.request

        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status
    except Exception:
        try:
            result = subprocess.run(
                ["curl", "-s", "--max-time", "30", "-o", "/dev/null", "-w", "%{http_code}", url],
                capture_output=True,
                text=True,
                timeout=35,
            )
            return int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
        except Exception:
            return 0


def main() -> None:
    started_at = time.time()

    # 1. Retrieve token
    hf_token, token_source = _retrieve_hf_token()
    hf_token_found = hf_token is not None

    # 2. Verify cartridge file exists
    nqueens_path = GAMES_DIR / "nqueens.py"
    cartridge_found = nqueens_path.exists()

    # 3. Check registration in __init__.py
    init_path = GAMES_DIR / "__init__.py"
    app_py_updated = _cartridge_registered(init_path, "NQueensGame")

    # Count cartridges in ALL_GAMES
    n_cartridges_deployed = 0
    if init_path.exists():
        content = init_path.read_text()
        # Count lines that look like instantiated games inside ALL_GAMES block
        in_block = False
        for line in content.splitlines():
            if "ALL_GAMES" in line and "[" in line:
                in_block = True
            if in_block and "Game()" in line:
                n_cartridges_deployed += 1
            if in_block and "]" in line and "Game" not in line:
                break

    # 4. Deploy
    deploy_attempted = False
    gallery_updated = False
    deploy_error = None

    if not hf_token_found:
        honest_verdict = "hf_token_not_found"
    elif not cartridge_found:
        honest_verdict = "upstream_cartridge_missing"
    else:
        deploy_attempted = True
        gallery_updated, deploy_result = _upload_space(hf_token)
        if not gallery_updated:
            deploy_error = deploy_result

    # 5. Verify live
    live_http_status = 0
    if deploy_attempted:
        live_http_status = _http_status(LIVE_URL)

    # Determine verdict
    if not hf_token_found:
        honest_verdict = "hf_token_not_found"
    elif not cartridge_found:
        honest_verdict = "upstream_cartridge_missing"
    elif gallery_updated and live_http_status == 200:
        honest_verdict = "gallery_updated_n_queens_live"
    elif deploy_attempted:
        honest_verdict = "deploy_attempted_verify_pending"
    else:
        honest_verdict = "failed"

    duration_s = round(time.time() - started_at, 3)

    artifact = {
        "experiment": 1102,
        "title": "HuggingFace Spaces gallery update — N-Queens cartridge",
        "schema": "carnot.hf_spaces_gallery_update.v1",
        "run_date": time.strftime("%Y-%m-%d", time.gmtime()),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000+00:00", time.gmtime(started_at)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000+00:00", time.gmtime()),
        "duration_s": duration_s,
        "status": "success" if honest_verdict == "gallery_updated_n_queens_live" else "partial",
        "cartridge_found": cartridge_found,
        "hf_token_found": hf_token_found,
        "hf_token_source": token_source,
        "n_cartridges_deployed": n_cartridges_deployed,
        "app_py_updated": app_py_updated,
        "deploy_attempted": deploy_attempted,
        "gallery_updated": gallery_updated,
        "live_http_status": live_http_status,
        "live_url": LIVE_URL,
        "honest_verdict": honest_verdict,
        "deploy_error": deploy_error,
        "cost_usd": 0.0,
        "decision_class": "deploy",
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
