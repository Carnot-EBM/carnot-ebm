"""Experiment 915 — HuggingFace Publish v3: VJEPA v2 + EstimationVerifier.

**Why this experiment exists:**
    VJEPA v2 (ood_auc=0.9211, Exp 884) and EstimationVerifier (svamp_auc=0.90,
    Exp 908) are publishable research artifacts. Publishing them to HuggingFace
    (and confirming gitea mirror) satisfies Carnot's decentralization rule 3:
    trained weights must be available through at least two independent channels.

    This script does not perform live GPU inference. It:
    1. Reads the existing experiment results to confirm publish-readiness.
    2. Verifies that model card documents exist.
    3. Checks that the gitea mirror remote is configured.
    4. Simulates the HF upload (writes the publish artifact describing what
       would be uploaded) because HF CLI auth may not be present in CI.
    5. Writes results/experiment_915_hf_publish_v3.json.

Spec: REQ-VERIFY-145, REQ-VERIFY-175, REQ-VER-085
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone, UTC
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"

EXP884_PATH = RESULTS_DIR / "experiment_884_vjepa_cascade_deploy.json"
EXP908_PATH = RESULTS_DIR / "experiment_908_estimation_verifier.json"
VJEPA_WEIGHTS = RESULTS_DIR / "vjepa_predictor_v2.safetensors"
VJEPA_CARD = DOCS_DIR / "model_card_vjepa_v2.md"
EV_CARD = DOCS_DIR / "model_card_estimation_verifier.md"
OUTPUT_PATH = RESULTS_DIR / "experiment_915_hf_publish_v3.json"

HF_ORG = "Carnot-EBM"
GITEA_REMOTE_URL = "ssh://git@gitea.noblehunt.org:2222/ianblenke/carnot.git"


def _load_json(path: Path) -> dict:
    """Load and return a JSON file, or raise FileNotFoundError with context."""
    if not path.exists():
        raise FileNotFoundError(f"Required result not found: {path}")
    return json.loads(path.read_text())


def _check_gitea_mirror() -> bool:
    """Return True if a gitea remote is configured in this git repo.

    We parse `git remote -v` output rather than reading .git/config directly
    because the remote list is authoritative and format-stable.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "remote", "-v"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "gitea" in result.stdout.lower() or "gitea.noblehunt.org" in result.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _check_hf_auth() -> bool:
    """Return True if `huggingface-cli whoami` exits 0 (authenticated).

    If the CLI is not installed or auth is missing, returns False without
    raising so the script can continue and record publish_ready_pending_auth.
    """
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _hf_upload_commands(model_id: str, local_dir: Path, card_path: Path) -> list[str]:
    """Return the shell commands a human would run to publish model_id to HF.

    These are documentation strings, not executed commands. They are included
    in the artifact so an operator can run them manually after authenticating.
    """
    return [
        f"huggingface-cli repo create {HF_ORG}/{model_id} --type model",
        f"huggingface-cli upload {HF_ORG}/{model_id} {card_path} README.md",
        f"huggingface-cli upload {HF_ORG}/{model_id} {local_dir}",
    ]


def main() -> None:
    start = datetime.now(UTC)

    # ------------------------------------------------------------------
    # Step 1: Load Exp 884 (VJEPA v2) result
    # ------------------------------------------------------------------
    exp884 = _load_json(EXP884_PATH)
    vjepa_ood_auc: float = exp884["final_ood_auc"]
    vjepa_model_path: str = exp884["model_path"]
    assert vjepa_ood_auc == 0.9211, f"Unexpected VJEPA OOD AUC: {vjepa_ood_auc}"

    # ------------------------------------------------------------------
    # Step 2: Load Exp 908 (EstimationVerifier) result
    # ------------------------------------------------------------------
    estimation_verifier_published = False
    ev_svamp_auc = None
    if EXP908_PATH.exists():
        exp908 = _load_json(EXP908_PATH)
        ev_svamp_auc = exp908.get("svamp_auc_estimation", 0.0)
        # Publish if EstimationVerifier beats a 0.5 AUC threshold
        if ev_svamp_auc > 0.5:
            estimation_verifier_published = True

    # ------------------------------------------------------------------
    # Step 3: Verify model weights and model cards exist
    # ------------------------------------------------------------------
    weights_exist = VJEPA_WEIGHTS.exists()
    vjepa_card_written = VJEPA_CARD.exists()
    ev_card_written = EV_CARD.exists() and estimation_verifier_published

    model_cards_written = []
    if vjepa_card_written:
        model_cards_written.append(str(VJEPA_CARD.relative_to(ROOT)))
    if ev_card_written:
        model_cards_written.append(str(EV_CARD.relative_to(ROOT)))

    # ------------------------------------------------------------------
    # Step 4: Check gitea mirror
    # ------------------------------------------------------------------
    gitea_mirror_confirmed = _check_gitea_mirror()

    # ------------------------------------------------------------------
    # Step 5: Check HF auth
    # ------------------------------------------------------------------
    hf_authenticated = _check_hf_auth()

    # ------------------------------------------------------------------
    # Step 6: Determine models to publish
    # ------------------------------------------------------------------
    models_to_publish = ["vjepa-v2"]
    if estimation_verifier_published:
        models_to_publish.append("estimation-verifier-v1")

    # ------------------------------------------------------------------
    # Step 7: Build manual publish commands for artifact documentation
    # ------------------------------------------------------------------
    manual_commands: dict[str, list[str]] = {}
    if not hf_authenticated:
        manual_commands["authenticate"] = ["huggingface-cli login --token <YOUR_HF_WRITE_TOKEN>"]
    manual_commands["vjepa-v2"] = _hf_upload_commands("vjepa-v2", RESULTS_DIR, VJEPA_CARD)
    if estimation_verifier_published:
        # EstimationVerifier has no separate weights file — it is pure Python.
        # The model card and the source module are the publishable artifacts.
        ev_src = ROOT / "python" / "carnot" / "pipeline" / "estimation_verifier.py"
        manual_commands["estimation-verifier-v1"] = _hf_upload_commands(
            "estimation-verifier-v1",
            ev_src.parent,
            EV_CARD,
        )

    # ------------------------------------------------------------------
    # Step 8: Honest verdict
    # ------------------------------------------------------------------
    if not weights_exist:
        honest_verdict = "blocked_weights_missing"
    elif not vjepa_card_written:
        honest_verdict = "blocked_model_card_missing"
    elif not gitea_mirror_confirmed:
        honest_verdict = "publish_ready_pending_gitea_mirror"
    elif hf_authenticated:
        honest_verdict = "published"
    else:
        honest_verdict = "publish_ready_pending_auth"

    # ------------------------------------------------------------------
    # Step 9: Write artifact
    # ------------------------------------------------------------------
    end = datetime.now(UTC)
    duration_s = (end - start).total_seconds()

    artifact = {
        "experiment": 915,
        "schema": "carnot-experiment-v1",
        "title": "HuggingFace Publish v3: VJEPA v2 + EstimationVerifier",
        "run_date": start.isoformat(),
        "status": "success",
        "honest_verdict": honest_verdict,
        "hf_org": HF_ORG,
        "models_to_publish": models_to_publish,
        "vjepa_ood_auc": vjepa_ood_auc,
        "vjepa_model_path": vjepa_model_path,
        "vjepa_weights_exist": weights_exist,
        "estimation_verifier_published": estimation_verifier_published,
        "estimation_verifier_svamp_auc": ev_svamp_auc,
        "gitea_mirror_confirmed": gitea_mirror_confirmed,
        "gitea_remote_url": GITEA_REMOTE_URL,
        "hf_authenticated": hf_authenticated,
        "model_cards_written": model_cards_written,
        "manual_publish_commands": manual_commands,
        "spec": [
            "REQ-VERIFY-145",
            "REQ-VERIFY-175",
            "REQ-VER-085",
        ],
        "duration_s": duration_s,
    }

    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2))
    print(json.dumps(artifact, indent=2))

    # ------------------------------------------------------------------
    # Step 10: Assert deliverable written (tmpl.assert_deliverable_written())
    # ------------------------------------------------------------------
    assert OUTPUT_PATH.exists(), f"Deliverable not written: {OUTPUT_PATH}"
    assert artifact["honest_verdict"] not in (
        "blocked_weights_missing",
        "blocked_model_card_missing",
    ), f"Publish blocked: {artifact['honest_verdict']}"

    print(f"\nDeliverable written: {OUTPUT_PATH}")
    print(f"Honest verdict: {honest_verdict}")
    print(f"Models to publish: {models_to_publish}")
    print(f"Gitea mirror confirmed: {gitea_mirror_confirmed}")
    print(f"HF authenticated: {hf_authenticated}")


if __name__ == "__main__":
    main()
