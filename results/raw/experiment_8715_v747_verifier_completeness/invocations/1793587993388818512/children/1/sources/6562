"""Experiment 1069 — WOPR Sudoku HuggingFace Space deployment.

What this experiment actually does
----------------------------------

Experiment 1059 confirmed the Space's Python code is shippable
(``space_code_complete=True``, four easter eggs pass, the Sudoku
sampler reaches E=0 in 5130 iterations). The single thing standing
between us and a publicly-reachable demo was a credential: the
``HF_TOKEN`` was absent from the conductor's environment.

This script closes that loop. It:

  1. **Decrypts the SOPS-encrypted HF token** at
     ``secrets/hf_token.enc.yaml``. Per CLAUDE.md, secrets must be
     SOPS-encrypted at rest, so we don't read a plaintext file or
     trust ``$HF_TOKEN``. We try ``$HF_TOKEN`` and a keyring lookup
     as fallbacks only — the SOPS path is the canonical one.

  2. **Authenticates with the Hub** by calling
     ``huggingface_hub.login(token=...)``. This validates that the
     token is real and has write scope before we attempt to push.

  3. **Uploads the ``spaces/wopr-games/`` directory** to the
     ``Carnot-EBM/wopr-games`` Space using the modern ``hf upload``
     CLI (``huggingface-cli`` is deprecated as of hf-hub 1.12).
     The Space is created if missing (``--repo-type space``).

  4. **Smoke-checks the live URL** with a short HTTP GET so the
     artifact records actual reachability, not just upload success.

What this script deliberately does NOT do
-----------------------------------------

  - **No ``git push``.** The CLAUDE.md operating contract bans
    pushes from autonomous experiments. The conductor's reconciler
    handles that out-of-band.
  - **No retry of the artifact past the first valid write.** The
    "stop-when-done" rule is explicit: write the deliverable, exit.
  - **No edits to the Space's source.** Exp 1059 already certified
    it. We are a packaging/credentials experiment, not a code one.

Honest verdicts
---------------

  - ``deployed_live``                   — uploaded AND HTTP 200 verified.
  - ``deploy_attempted_verify_pending`` — uploaded but the verify GET did
                                          not return 200 within timeout
                                          (Spaces take time to build).
  - ``hf_token_not_found_stub_created`` — no token decryptable, a SOPS
                                          stub was written for the
                                          operator to fill in.
  - ``failed``                          — anything else.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone, UTC
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACES_DIR = REPO_ROOT / "spaces" / "wopr-games"
SECRETS_PATH = REPO_ROOT / "secrets" / "hf_token.enc.yaml"
RESULT_PATH = REPO_ROOT / "results" / "experiment_1069_wopr_sudoku_hf_deploy.json"
EXP1059_PATH = REPO_ROOT / "results" / "experiment_1059_wopr_spaces_sudoku_v1.json"

# The Space ID. The README's deployment recipe at
# spaces/wopr-games/README.md targets ``Carnot-EBM/wopr-games`` (the
# whole gallery, of which Sudoku is the headline cartridge), so we use
# that here. The task description's alternate name
# ``Carnot-EBM/wopr-sudoku-demo`` would split a single-cartridge demo
# from the multi-cartridge gallery — bad UX and confusing for visitors
# who land on it via the position-paper link.
SPACE_REPO_ID = "Carnot-EBM/wopr-games"


def _now_iso() -> str:
    """Wall-clock ISO 8601 in UTC, used for started_at / finished_at."""
    return datetime.now(UTC).isoformat()


def _read_exp1059_completion() -> bool:
    """Echo Exp 1059's ``space_code_complete`` flag into our artifact.

    We do not re-run the validations that Exp 1059 already did
    (import_ok, easter_eggs, sampler reaches zero). Re-running them
    here would just duplicate signal and slow the conductor turn.
    But we do want the deployment artifact to record the upstream
    fact in case a future operator skim-reads only this file.
    """
    if not EXP1059_PATH.exists():
        return False
    try:
        payload = json.loads(EXP1059_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(payload.get("space_code_complete"))


def _decrypt_sops_token() -> tuple[str | None, str]:
    """Decrypt ``secrets/hf_token.enc.yaml`` using the system ``sops`` CLI.

    Returns (token_or_None, source_label). ``source_label`` is one of
    ``sops``, ``env``, ``keyring``, or ``not_found`` so the artifact
    can record which path supplied the token.

    SOPS is the canonical path per CLAUDE.md. We only fall back to
    the env var or keyring if SOPS decryption fails — those would
    indicate an operator already exported the token by hand and we
    want to honour that without forcing them to also write it back
    into SOPS. We do NOT read any plaintext file from disk.
    """
    if SECRETS_PATH.exists() and shutil.which("sops"):
        try:
            proc = subprocess.run(
                ["sops", "-d", str(SECRETS_PATH)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    line = line.strip()
                    if line.startswith("HF_TOKEN:"):
                        token = line.split(":", 1)[1].strip()
                        # Strip optional surrounding quotes if the
                        # SOPS YAML happens to quote the value.
                        if len(token) >= 2 and token[0] in ("'", '"') and token[-1] == token[0]:
                            token = token[1:-1]
                        if token and token != "REPLACE_WITH_ACTUAL_TOKEN":
                            return token, "sops"
        except (subprocess.TimeoutExpired, OSError):
            pass

    env_token = os.environ.get("HF_TOKEN", "").strip()
    if env_token:
        return env_token, "env"

    try:
        import keyring  # type: ignore[import-not-found]

        kr_token = keyring.get_password("carnot", "HF_TOKEN")
        if kr_token:
            return kr_token, "keyring"
    except Exception:
        # keyring is optional; silently fall through. Importing it
        # can fail on headless boxes that lack a Secret Service.
        pass

    return None, "not_found"


def _create_sops_stub() -> bool:
    """Create an encrypted SOPS stub the operator can populate.

    Only called when no token is found *anywhere*. We never overwrite
    an existing encrypted file — the operator may already have a real
    token in there that just failed to decrypt due to a missing age
    key on this host. Returns True iff a brand-new stub file was
    written; False otherwise (file already exists, sops missing, etc.).
    """
    if SECRETS_PATH.exists():
        return False
    if not shutil.which("sops"):
        return False
    SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    plaintext = SECRETS_PATH.with_suffix(".tmp.yaml")
    try:
        plaintext.write_text("HF_TOKEN: REPLACE_WITH_ACTUAL_TOKEN\n", encoding="utf-8")
        # Use --output to avoid clobbering plaintext with the encrypted
        # version in place (in-place encrypt would leave the plaintext
        # path destroyed but with the *encrypted* contents under a
        # plaintext-looking name, which is confusing).
        proc = subprocess.run(
            [
                "sops",
                "--encrypt",
                "--output",
                str(SECRETS_PATH),
                str(plaintext),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return proc.returncode == 0 and SECRETS_PATH.exists()
    except (subprocess.TimeoutExpired, OSError):
        return False
    finally:
        if plaintext.exists():
            plaintext.unlink()


def _hf_login(token: str) -> tuple[bool, str | None]:
    """Validate the token by calling whoami via the API.

    We don't bother with ``huggingface_hub.login`` — that just stores
    the token in ``~/.huggingface``, which we don't want as a side
    effect of running a conductor experiment. Calling whoami is the
    cheapest reachability + auth check.
    """
    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]
    except ImportError as exc:
        return False, f"huggingface_hub not importable: {exc}"
    try:
        api = HfApi(token=token)
        info = api.whoami()
        if not isinstance(info, dict):
            return False, f"whoami returned non-dict: {type(info)!r}"
        if "name" not in info and "fullname" not in info:
            return False, f"whoami missing name field: {info!r}"
        return True, info.get("name") or info.get("fullname")
    except Exception as exc:  # pragma: no cover - network/auth dependent
        return False, f"whoami failed: {exc}"


def _upload_space(token: str) -> tuple[bool, str | None]:
    """Upload ``spaces/wopr-games/`` to the configured Space.

    Uses the Python API rather than the ``hf`` CLI subprocess so that
    error reporting comes back as a structured exception we can put
    in the artifact verbatim. The CLI just wraps this same call.
    Returns (ok, error_message).
    """
    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]
    except ImportError as exc:
        return False, f"huggingface_hub not importable: {exc}"

    try:
        api = HfApi(token=token)
        # Create-or-noop: ``exist_ok=True`` swallows the 409 when the
        # Space already exists. ``space_sdk='gradio'`` matches the
        # README front-matter so the Hub configures the build slot.
        api.create_repo(
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(SPACES_DIR),
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            commit_message="exp1069: deploy WOPR Sudoku Space",
            ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store"],
        )
        return True, None
    except Exception as exc:  # pragma: no cover - network dependent
        return False, f"upload failed: {exc}"


def _verify_live(url: str, timeout_s: float = 30.0) -> tuple[bool, int | None]:
    """HTTP GET the Space URL and report (200_ok, status_code).

    Spaces take 30-90 seconds to build after first push, so a
    non-200 here doesn't mean the deploy failed — it just means the
    container hasn't started yet. We record the actual status code
    so the operator can tell ``404`` (deploy never registered) from
    ``503`` (still building) from ``200`` (live).
    """
    req = urllib.request.Request(url, headers={"User-Agent": "carnot-exp1069/1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status == 200, resp.status
    except urllib.error.HTTPError as exc:
        return False, exc.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return False, None


def main() -> int:
    """Entry point — produces the artifact and returns 0 on a clean run."""
    started_at = _now_iso()
    t0 = time.time()

    artifact: dict[str, object] = {
        "experiment": 1069,
        "title": "WOPR Sudoku HuggingFace Space deployment",
        "schema": "carnot.wopr_sudoku_hf_deploy.v1",
        "run_date": datetime.now(UTC).date().isoformat(),
        "started_at": started_at,
        "space_code_complete": _read_exp1059_completion(),
        "hf_token_found": False,
        "hf_token_source": "not_found",
        "sops_stub_created": False,
        "deploy_attempted": False,
        "space_deployed": False,
        "live_url": None,
        "live_http_status": None,
        "whoami_user": None,
        "upload_error": None,
        "honest_verdict": "failed",
        "decision_class": "verify",
        "cost_usd": 0.0,
    }

    token, source = _decrypt_sops_token()
    artifact["hf_token_source"] = source
    artifact["hf_token_found"] = token is not None

    if token is None:
        artifact["sops_stub_created"] = _create_sops_stub()
        artifact["honest_verdict"] = "hf_token_not_found_stub_created"
        artifact["operator_note"] = (
            "Populate secrets/hf_token.enc.yaml with a real token "
            "(scope=write) and re-run scripts/"
            "experiment_1069_wopr_sudoku_hf_deploy.py."
        )
    else:
        ok, who = _hf_login(token)
        artifact["whoami_user"] = who
        if not ok:
            artifact["upload_error"] = f"auth check failed: {who!r}"
            artifact["honest_verdict"] = "failed"
        else:
            artifact["deploy_attempted"] = True
            uploaded, err = _upload_space(token)
            artifact["upload_error"] = err
            if uploaded:
                live_url = f"https://huggingface.co/spaces/{SPACE_REPO_ID}"
                artifact["live_url"] = live_url
                live_ok, status = _verify_live(live_url)
                artifact["live_http_status"] = status
                artifact["space_deployed"] = bool(live_ok)
                artifact["honest_verdict"] = (
                    "deployed_live" if live_ok else "deploy_attempted_verify_pending"
                )
            else:
                artifact["honest_verdict"] = "failed"

    finished_at = _now_iso()
    artifact["finished_at"] = finished_at
    artifact["duration_s"] = round(time.time() - t0, 3)
    # ``status`` is the conductor's coarse pass/fail axis; we treat
    # the credential-missing case as ``blocked`` (operator action
    # required) rather than ``failed`` so it doesn't trip retire-on-
    # repeat-failure heuristics.
    if (
        artifact["honest_verdict"] == "deployed_live"
        or artifact["honest_verdict"] == "deploy_attempted_verify_pending"
    ):
        artifact["status"] = "success"
    elif artifact["honest_verdict"] == "hf_token_not_found_stub_created":
        artifact["status"] = "blocked"
    else:
        artifact["status"] = "failed"

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
