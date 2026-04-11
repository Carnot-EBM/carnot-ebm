"""Experiment 164 — HuggingFace Publishing: guided-decoding-adapter, constraint-propagation
models, JEPA predictor v2, and per-token EBM README updates.

Spec refs: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, NFR-03
Story: HuggingFace Publishing Milestones (research-program.md)

What this script does
---------------------
1. Installs huggingface_hub if not present (in the venv).
2. Checks for HuggingFace authentication via the hub token.
3. Uploads exports/guided-decoding-adapter → Carnot-EBM/guided-decoding-adapter
4. Uploads exports/constraint-propagation-models/(arithmetic|logic|code) →
   Carnot-EBM/constraint-propagation-(domain)
5. Creates a model card for the JEPA predictor v2 (results/jepa_predictor_v2.safetensors,
   trained in Exp 155) and uploads it to Carnot-EBM/jepa-predictor-v2.
6. Verifies each upload by downloading the README back from the hub.
7. Updates the 16 per-token EBM model READMEs on HuggingFace to add a
   "pip install carnot" note.
8. Writes results/experiment_164_results.json.
9. If not authenticated, writes all commands to scripts/hf_upload_commands.sh
   and exits gracefully (non-zero exit code indicates dry-run mode).
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = REPO_ROOT / "exports" / "guided-decoding-adapter"
CONSTRAINT_DIR = REPO_ROOT / "exports" / "constraint-propagation-models"
JEPA_V2_WEIGHTS = REPO_ROOT / "results" / "jepa_predictor_v2.safetensors"
JEPA_EXP155_RESULTS = REPO_ROOT / "results" / "experiment_155_results.json"
RESULTS_PATH = REPO_ROOT / "results" / "experiment_164_results.json"
UPLOAD_SCRIPT_PATH = REPO_ROOT / "scripts" / "hf_upload_commands.sh"
JEPA_CARD_DIR = REPO_ROOT / "results" / "jepa_predictor_v2_card"

HF_ORG = "Carnot-EBM"
CONSTRAINT_DOMAINS = ["arithmetic", "logic", "code"]

# The 16 existing per-token EBM model repos on Carnot-EBM (from Exp 150).
# These are Phase 1 research artifacts that need a pip install note added
# so visitors know how to use the framework.
PER_TOKEN_MODELS: list[str] = [
    "per-token-ebm-bonsai-17b-nothink",
    "per-token-ebm-gemma4-e2b-it-nothink",
    "per-token-ebm-gemma4-e2b-nothink",
    "per-token-ebm-gemma4-e4b-it-nothink",
    "per-token-ebm-gemma4-e4b-nothink",
    "per-token-ebm-gptoss-20b-nothink",
    "per-token-ebm-lfm25-12b-nothink",
    "per-token-ebm-lfm25-350m-nothink",
    "per-token-ebm-qwen3-06b",
    "per-token-ebm-qwen35-08b-nothink",
    "per-token-ebm-qwen35-08b-think",
    "per-token-ebm-qwen35-27b-nothink",
    "per-token-ebm-qwen35-2b-nothink",
    "per-token-ebm-qwen35-35b-nothink",
    "per-token-ebm-qwen35-4b-nothink",
    "per-token-ebm-qwen35-9b-nothink",
]

# The note to prepend to each per-token EBM README.
# We add it after the YAML front-matter block (between --- delimiters).
PIP_NOTE = textwrap.dedent("""\
    > **Note:** These are Phase 1 research artifacts — per-token activation EBMs
    > that detect hallucination confidence signals from LLM hidden states. For
    > production use of the full Carnot EBM framework (constraint verification,
    > guided decoding, energy-based repair), see:
    >
    > ```bash
    > pip install carnot
    > ```
    >
    > Source and documentation: <https://github.com/ianblenke/carnot>

""")


# ---------------------------------------------------------------------------
# Step 0 — Ensure huggingface_hub is available
# ---------------------------------------------------------------------------
def ensure_huggingface_hub() -> None:
    """Install huggingface_hub into the active venv if it is not importable."""
    try:
        import huggingface_hub  # noqa: F401
        print("[HF] huggingface_hub already installed.")
    except ImportError:
        print("[HF] Installing huggingface_hub …")
        venv_pip = REPO_ROOT / ".venv" / "bin" / "pip"
        pip_cmd = str(venv_pip) if venv_pip.exists() else sys.executable + " -m pip"
        subprocess.check_call(
            [pip_cmd, "install", "--quiet", "huggingface_hub"],
            shell=isinstance(pip_cmd, str),
        )
        import huggingface_hub  # noqa: F401 — re-check after install
        print("[HF] huggingface_hub installed successfully.")


# ---------------------------------------------------------------------------
# Step 1 — Check authentication
# ---------------------------------------------------------------------------
def check_authenticated() -> tuple[bool, str]:
    """Return (authenticated, username_or_reason).

    Uses huggingface_hub.whoami() which reads the cached token from
    ~/.cache/huggingface/token or the HF_TOKEN env var.
    """
    from huggingface_hub import whoami
    from huggingface_hub.errors import LocalTokenNotFoundError

    try:
        info = whoami()
        username = info.get("name", "unknown")
        print(f"[HF] Authenticated as: {username}")
        return True, username
    except LocalTokenNotFoundError:
        return False, "No HuggingFace token found (run `huggingface-cli login` or set HF_TOKEN)"
    except Exception as exc:  # network errors, etc.
        return False, f"Authentication check failed: {exc}"


# ---------------------------------------------------------------------------
# Step 2 — Upload guided-decoding-adapter (Exp 137)
# ---------------------------------------------------------------------------
def upload_guided_decoding_adapter(api) -> dict:
    """Upload exports/guided-decoding-adapter → Carnot-EBM/guided-decoding-adapter."""
    repo_id = f"{HF_ORG}/guided-decoding-adapter"
    print(f"[HF] Uploading {ADAPTER_DIR} → {repo_id} …")

    try:
        # Ensure the repo exists (creates if absent; no-op if already there).
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        url = api.upload_folder(
            folder_path=str(ADAPTER_DIR),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Exp 164: upload guided-decoding-adapter (Exp 137 artifact)",
        )
        print(f"[HF] ✓ Uploaded guided-decoding-adapter → {url}")
        return {"repo_id": repo_id, "status": "uploaded", "url": str(url)}
    except Exception as exc:
        print(f"[HF] ✗ Upload failed: {exc}")
        return {"repo_id": repo_id, "status": "failed", "error": str(exc)}


# ---------------------------------------------------------------------------
# Step 3 — Upload constraint propagation models (Exp 151)
# ---------------------------------------------------------------------------
def upload_constraint_models(api) -> list[dict]:
    """Upload each domain sub-directory of constraint-propagation-models."""
    results = []
    for domain in CONSTRAINT_DOMAINS:
        domain_dir = CONSTRAINT_DIR / domain
        repo_id = f"{HF_ORG}/constraint-propagation-{domain}"
        print(f"[HF] Uploading {domain_dir} → {repo_id} …")

        if not domain_dir.exists():
            print(f"[HF] ✗ Directory not found: {domain_dir}")
            results.append({"repo_id": repo_id, "status": "skipped", "reason": "directory not found"})
            continue

        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
            url = api.upload_folder(
                folder_path=str(domain_dir),
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Exp 164: upload constraint-propagation-{domain} (Exp 151 artifact)",
            )
            print(f"[HF] ✓ Uploaded {domain} → {url}")
            results.append({"repo_id": repo_id, "status": "uploaded", "url": str(url)})
        except Exception as exc:
            print(f"[HF] ✗ Upload failed for {domain}: {exc}")
            results.append({"repo_id": repo_id, "status": "failed", "error": str(exc)})

    return results


# ---------------------------------------------------------------------------
# Step 4 — Build JEPA predictor v2 model card and upload (Exp 155)
# ---------------------------------------------------------------------------
def _build_jepa_card(exp155: dict) -> str:
    """Return the README.md content for the JEPA predictor v2 model card."""
    v2_auroc = exp155.get("v2_macro_auroc", 0.659)
    per_domain = exp155.get("per_domain_v2", {})
    arith = per_domain.get("arithmetic", 0.721)
    code = per_domain.get("code", 0.776)
    logic = per_domain.get("logic", 0.479)
    best_epoch = exp155.get("training", {}).get("best_epoch", 19)
    n_train = exp155.get("training", {}).get("n_train", 963)

    return textwrap.dedent(f"""\
        ---
        tags:
          - energy-based-model
          - jepa
          - constraint-verification
          - carnot
          - jax
        license: apache-2.0
        ---

        # jepa-predictor-v2

        **JEPA Violation Predictor v2** — a multi-domain neural verifier trained to predict
        whether a language model answer violates domain constraints (arithmetic, code, logic).

        This is a research artifact from the [Carnot EBM framework](https://github.com/ianblenke/carnot).
        It is a small MLP (input_dim=200, hidden_dim=128, output_dim=1) trained with class-weighted
        binary cross-entropy and early stopping on a balanced multi-domain dataset.

        ## Performance (Exp 155)

        | Domain | AUROC (v2) | vs v1 |
        |--------|-----------|-------|
        | Arithmetic | {arith:.3f} | +0.018 |
        | Code | {code:.3f} | +0.071 |
        | Logic | {logic:.3f} | −0.056 |
        | **Macro average** | **{v2_auroc:.3f}** | **+0.011** |

        Training details: {n_train} samples, best epoch {best_epoch} / 100, early stopping on
        validation macro AUROC (patience=15), class-weighted BCE loss (pos_weight clipped [0.1, 10]).

        ## Architecture

        ```
        JEPAViolationPredictor(
          input_dim   = 200    # 200-dim binary feature vector (same encoder as Ising models)
          hidden_dim  = 128
          output_dim  = 1      # P(violation | x)
        )
        ```

        The predictor is trained on structured (question, answer, violated) triples. Features are
        the same 200-dim binary structural encoding used by the Ising constraint models
        (Carnot-EBM/constraint-propagation-*). This makes the two model families interoperable.

        ## Training Data (Exp 154 → Exp 155)

        - **Arithmetic**: 800 pairs reused from Exp 143 (carry-chain arithmetic templates)
        - **Code**: 200 synthetic pairs (Python type, return, initialisation constraints)
        - **Logic**: 200 synthetic pairs (implication, exclusion, disjunction, negation)
        - Total: 1,200 pairs; 963 train / 237 validation (stratified by domain × violated)

        ## Usage

        ```python
        from safetensors.numpy import load_file

        weights = load_file("jepa_predictor_v2.safetensors")
        # Keys: "layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias"

        # Or load via the carnot package:
        from carnot.inference.jepa_predictor import JEPAViolationPredictor
        predictor = JEPAViolationPredictor.load("jepa_predictor_v2.safetensors")

        # Score a 200-dim binary feature vector
        import numpy as np
        x = np.zeros(200, dtype=np.float32)
        x[:20] = 1.0   # example features
        prob_violation = predictor.predict(x)
        print(f"P(violation) = {{prob_violation:.3f}}")
        ```

        ## Limitations

        1. **Logic AUROC low (0.479)**: byte-level structural features do not capture
           logical implication structure well. Semantic embeddings are needed.
        2. **Code domain fast-path issues** (Exp 156): at threshold ≥ 0.3, all code questions
           are fast-pathed (200/200), causing degradation. Use threshold ≥ 0.8 for code.
        3. **Target not met**: <2% degradation target at t=0.5 not achieved. Treat as
           directional baseline, not production-ready.
        4. **Template training data**: Not validated on real LLM outputs.

        ## Note: Production Installation

        > **Note:** This is a Phase 1 research artifact. For production use of the full
        > Carnot EBM framework (constraint verification, guided decoding, energy-based repair), see:
        >
        > ```bash
        > pip install carnot
        > ```
        >
        > Source and documentation: <https://github.com/ianblenke/carnot>

        ## Spec

        - REQ-JEPA-001: Violation predictor trained on multi-domain data.
        - REQ-JEPA-002: Fast-path routing using predictor confidence.
        - SCENARIO-JEPA-003: Cross-domain AUROC improvement over v1.

        ## Citation

        ```bibtex
        @misc{{carnot2026jepa,
          title  = {{Carnot JEPA Violation Predictor v2}},
          author = {{Carnot-EBM}},
          year   = {{2026}},
          url    = {{https://github.com/ianblenke/carnot}}
        }}
        ```
    """)


def upload_jepa_v2(api) -> dict:
    """Create model card dir, copy weights, upload to Carnot-EBM/jepa-predictor-v2."""
    repo_id = f"{HF_ORG}/jepa-predictor-v2"
    print(f"[HF] Uploading JEPA predictor v2 → {repo_id} …")

    if not JEPA_V2_WEIGHTS.exists():
        print(f"[HF] ✗ Weights not found: {JEPA_V2_WEIGHTS}")
        return {"repo_id": repo_id, "status": "skipped", "reason": "weights file not found"}

    # Load Exp 155 results for the model card
    exp155: dict = {}
    if JEPA_EXP155_RESULTS.exists():
        with open(JEPA_EXP155_RESULTS) as f:
            exp155 = json.load(f)

    # Build model card directory (temp, cleaned up after upload)
    JEPA_CARD_DIR.mkdir(parents=True, exist_ok=True)
    card_weights = JEPA_CARD_DIR / "jepa_predictor_v2.safetensors"
    card_readme = JEPA_CARD_DIR / "README.md"

    # Copy weights (do not modify the original)
    import shutil
    shutil.copy2(JEPA_V2_WEIGHTS, card_weights)

    # Write model card
    card_readme.write_text(_build_jepa_card(exp155), encoding="utf-8")
    print(f"[HF] Model card written to {card_readme}")

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        url = api.upload_folder(
            folder_path=str(JEPA_CARD_DIR),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Exp 164: upload JEPA violation predictor v2 (Exp 155 artifact)",
        )
        print(f"[HF] ✓ Uploaded JEPA v2 → {url}")
        return {"repo_id": repo_id, "status": "uploaded", "url": str(url)}
    except Exception as exc:
        print(f"[HF] ✗ Upload failed for JEPA v2: {exc}")
        return {"repo_id": repo_id, "status": "failed", "error": str(exc)}


# ---------------------------------------------------------------------------
# Step 5 — Verify uploads by downloading READMEs back
# ---------------------------------------------------------------------------
def verify_upload(api, repo_id: str) -> dict:
    """Download README.md from repo_id and confirm it is non-empty."""
    from huggingface_hub import hf_hub_download

    print(f"[HF] Verifying {repo_id} …")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type="model",
                local_dir=tmpdir,
                force_download=True,
            )
            size = Path(local_path).stat().st_size
            if size == 0:
                return {"repo_id": repo_id, "verify": "empty_readme", "size_bytes": 0}
            print(f"[HF] ✓ Verified {repo_id} — README.md {size} bytes")
            return {"repo_id": repo_id, "verify": "ok", "readme_bytes": size}
    except Exception as exc:
        print(f"[HF] ✗ Verify failed for {repo_id}: {exc}")
        return {"repo_id": repo_id, "verify": "failed", "error": str(exc)}


# ---------------------------------------------------------------------------
# Step 6 — Update per-token EBM READMEs with pip install note
# ---------------------------------------------------------------------------
def _prepend_pip_note(existing_readme: str) -> str:
    """Insert the pip install note after the YAML front-matter block.

    If the README starts with ---, find the closing --- and insert after it.
    Otherwise, prepend to the very top.
    """
    if not existing_readme.startswith("---"):
        return PIP_NOTE + existing_readme

    # Find closing ---
    closing = existing_readme.find("\n---", 3)
    if closing == -1:
        return PIP_NOTE + existing_readme

    # Position after the closing ---\n
    insert_at = closing + 4  # len("\n---") == 4, skip the newline after ---
    # Advance past the newline following ---
    if insert_at < len(existing_readme) and existing_readme[insert_at] == "\n":
        insert_at += 1

    # Only insert if the note is not already present
    if "pip install carnot" in existing_readme:
        return existing_readme  # already patched

    return existing_readme[:insert_at] + "\n" + PIP_NOTE + existing_readme[insert_at:]


def update_per_token_readmes(api) -> list[dict]:
    """Download each per-token EBM README, prepend the pip note, and re-upload."""
    from huggingface_hub import hf_hub_download

    results = []
    for model_name in PER_TOKEN_MODELS:
        repo_id = f"{HF_ORG}/{model_name}"
        print(f"[HF] Updating README for {repo_id} …")

        try:
            # Download current README
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="README.md",
                    repo_type="model",
                    local_dir=tmpdir,
                    force_download=True,
                )
                current_readme = Path(local_path).read_text(encoding="utf-8")

            # Check if the note is already present
            if "pip install carnot" in current_readme:
                print(f"[HF] ✓ {repo_id} already has pip note — skipped")
                results.append({"repo_id": repo_id, "status": "already_updated"})
                continue

            updated_readme = _prepend_pip_note(current_readme)

            # Upload updated README as a single file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(updated_readme)
                tmp_path = tmp.name

            try:
                api.upload_file(
                    path_or_fileobj=tmp_path,
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message="Exp 164: add pip install carnot note to model card",
                )
                print(f"[HF] ✓ Updated {repo_id}")
                results.append({"repo_id": repo_id, "status": "updated"})
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        except Exception as exc:
            print(f"[HF] ✗ Failed to update {repo_id}: {exc}")
            results.append({"repo_id": repo_id, "status": "failed", "error": str(exc)})

    return results


# ---------------------------------------------------------------------------
# Dry-run: write shell script with manual upload commands
# ---------------------------------------------------------------------------
def write_upload_script(reason: str) -> None:
    """Write scripts/hf_upload_commands.sh with all upload commands."""
    lines = [
        "#!/usr/bin/env bash",
        "# Experiment 164 — HuggingFace upload commands (generated by experiment_164_hf_publish.py)",
        f"# Reason for dry-run: {reason}",
        "# Run this after `huggingface-cli login` to publish all artifacts.",
        "",
        "set -euo pipefail",
        "",
        "# 1. Upload guided-decoding-adapter (Exp 137)",
        "huggingface-cli upload Carnot-EBM/guided-decoding-adapter \\",
        "    exports/guided-decoding-adapter \\",
        "    --repo-type model",
        "",
        "# 2. Upload constraint-propagation models (Exp 151)",
        *[
            f"huggingface-cli upload Carnot-EBM/constraint-propagation-{d} \\\n"
            f"    exports/constraint-propagation-models/{d} \\\n"
            "    --repo-type model"
            for d in CONSTRAINT_DOMAINS
        ],
        "",
        "# 3. Upload JEPA predictor v2 (Exp 155)",
        "# First run this script to generate the model card directory:",
        "# JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_164_hf_publish.py",
        "huggingface-cli upload Carnot-EBM/jepa-predictor-v2 \\",
        "    results/jepa_predictor_v2_card \\",
        "    --repo-type model",
        "",
        "# 4. Update per-token EBM READMEs (16 models)",
        "# These require a Python script since we need to download + patch + re-upload.",
        "# Re-run this script after authentication to apply the pip install note.",
        "",
        *[
            f"# huggingface-cli upload Carnot-EBM/{m} <patched-readme> --repo-type model"
            for m in PER_TOKEN_MODELS
        ],
    ]
    UPLOAD_SCRIPT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    UPLOAD_SCRIPT_PATH.chmod(0o755)
    print(f"[HF] Wrote upload commands to {UPLOAD_SCRIPT_PATH}")


# ---------------------------------------------------------------------------
# Write results JSON
# ---------------------------------------------------------------------------
def write_results(payload: dict) -> None:
    """Write results/experiment_164_results.json."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[HF] Results written to {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    """Return 0 on success, 1 on dry-run (unauthenticated)."""
    print("=== Experiment 164: HuggingFace Publishing ===")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()

    # ── Step 0: ensure the library is importable ──────────────────────────
    ensure_huggingface_hub()
    from huggingface_hub import HfApi

    # ── Step 1: check auth ────────────────────────────────────────────────
    authenticated, auth_info = check_authenticated()

    if not authenticated:
        print(f"[HF] Not authenticated: {auth_info}")
        print("[HF] Writing dry-run upload script instead …")

        # Still generate the JEPA model card directory (no network needed)
        exp155: dict = {}
        if JEPA_EXP155_RESULTS.exists():
            with open(JEPA_EXP155_RESULTS) as f:
                exp155 = json.load(f)
        JEPA_CARD_DIR.mkdir(parents=True, exist_ok=True)
        (JEPA_CARD_DIR / "README.md").write_text(
            _build_jepa_card(exp155), encoding="utf-8"
        )
        if JEPA_V2_WEIGHTS.exists():
            import shutil
            shutil.copy2(JEPA_V2_WEIGHTS, JEPA_CARD_DIR / "jepa_predictor_v2.safetensors")
            print(f"[HF] JEPA card dir prepared at {JEPA_CARD_DIR}")

        write_upload_script(auth_info)

        write_results(
            {
                "experiment": 164,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": "dry_run",
                "reason": auth_info,
                "upload_script": str(UPLOAD_SCRIPT_PATH),
                "jepa_card_dir": str(JEPA_CARD_DIR),
                "uploads": [],
                "readme_updates": [],
            }
        )
        print()
        print("[HF] Dry-run complete. Authenticate with `huggingface-cli login` then re-run.")
        return 1

    # ── Steps 2-5: authenticated path ────────────────────────────────────
    api = HfApi()

    # Upload guided-decoding-adapter
    adapter_result = upload_guided_decoding_adapter(api)
    adapter_verify = verify_upload(api, adapter_result["repo_id"])

    # Upload constraint propagation models
    constraint_results = upload_constraint_models(api)
    constraint_verifies = [
        verify_upload(api, r["repo_id"])
        for r in constraint_results
        if r.get("status") == "uploaded"
    ]

    # Upload JEPA predictor v2
    jepa_result = upload_jepa_v2(api)
    jepa_verify = (
        verify_upload(api, jepa_result["repo_id"])
        if jepa_result.get("status") == "uploaded"
        else {"repo_id": jepa_result["repo_id"], "verify": "skipped"}
    )

    # Update per-token EBM READMEs
    readme_update_results = update_per_token_readmes(api)

    # ── Write results ──────────────────────────────────────────────────────
    all_uploads = [adapter_result] + constraint_results + [jepa_result]
    all_verifies = [adapter_verify] + constraint_verifies + [jepa_verify]

    write_results(
        {
            "experiment": 164,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "live",
            "authenticated_as": auth_info,
            "uploads": [
                {**u, "verify": v}
                for u, v in zip(all_uploads, all_verifies)
            ] + [
                u for u in all_uploads[len(all_verifies):]  # any without verify
            ],
            "readme_updates": readme_update_results,
            "summary": {
                "total_uploads": len(all_uploads),
                "successful_uploads": sum(
                    1 for u in all_uploads if u.get("status") == "uploaded"
                ),
                "failed_uploads": sum(
                    1 for u in all_uploads if u.get("status") == "failed"
                ),
                "readme_updates_applied": sum(
                    1 for r in readme_update_results if r.get("status") == "updated"
                ),
                "readme_updates_already_done": sum(
                    1 for r in readme_update_results if r.get("status") == "already_updated"
                ),
                "readme_updates_failed": sum(
                    1 for r in readme_update_results if r.get("status") == "failed"
                ),
            },
        }
    )

    print()
    print("=== Experiment 164 complete ===")
    n_ok = sum(1 for u in all_uploads if u.get("status") == "uploaded")
    n_fail = sum(1 for u in all_uploads if u.get("status") == "failed")
    print(f"Uploads: {n_ok} succeeded, {n_fail} failed")
    n_readme_ok = sum(1 for r in readme_update_results if r.get("status") in ("updated", "already_updated"))
    print(f"README updates: {n_readme_ok}/{len(readme_update_results)} models processed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
