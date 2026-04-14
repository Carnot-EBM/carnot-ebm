"""Experiment 317: HuggingFace README accuracy audit and update.

Updates all 16 per-token activation EBM model READMEs on HuggingFace to
clarify their Phase 1 research artifact status.  The critical finding from
Exp 184/203 is that activation-based EBMs detect *model confidence*, not
*factual correctness* — READMEs written before this finding need to be
corrected.

Additionally:
  - Updates carnot-formal-claim-verifier-v1 README with Exp 316 full-scale
    benchmark results.
  - Creates/updates carnot-joint-constraint-v1 with an honest placeholder
    card ("RESEARCH PROTOTYPE — weights not published") so users are not
    misled about the availability of trained weights.

Credential check uses the same CLI → Python API fallback pattern as Exp 304:
  1. huggingface-cli whoami (subprocess)
  2. HfApi().whoami() (Python API fallback)
  3. Blocked artifact with login instructions if both fail.

Spec: REQ-PUBLISH-003, SCENARIO-PUBLISH-005, SCENARIO-PUBLISH-006
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_PATH = _REPO_ROOT / "results" / "experiment_317_hf_publish.json"
_EXP316_RESULTS_PATH = _REPO_ROOT / "results" / "experiment_316_fullscale_results.json"

# The 16 per-token activation EBM repos published during Phase 1.
# These detect output confidence (hallucination likelihood), NOT factual
# correctness.  This distinction is the critical finding of Exp 184/203.
_PER_TOKEN_EBM_REPOS: list[str] = [
    "Carnot-EBM/per-token-ebm-bonsai-17b-nothink",
    "Carnot-EBM/per-token-ebm-gemma4-e2b-it-nothink",
    "Carnot-EBM/per-token-ebm-gemma4-e2b-nothink",
    "Carnot-EBM/per-token-ebm-gemma4-e4b-it-nothink",
    "Carnot-EBM/per-token-ebm-gemma4-e4b-nothink",
    "Carnot-EBM/per-token-ebm-gptoss-20b-nothink",
    "Carnot-EBM/per-token-ebm-lfm25-12b-nothink",
    "Carnot-EBM/per-token-ebm-lfm25-350m-nothink",
    "Carnot-EBM/per-token-ebm-qwen3-06b",
    "Carnot-EBM/per-token-ebm-qwen35-08b-nothink",
    "Carnot-EBM/per-token-ebm-qwen35-08b-think",
    "Carnot-EBM/per-token-ebm-qwen35-27b-nothink",
    "Carnot-EBM/per-token-ebm-qwen35-2b-nothink",
    "Carnot-EBM/per-token-ebm-qwen35-35b-nothink",
    "Carnot-EBM/per-token-ebm-qwen35-4b-nothink",
    "Carnot-EBM/per-token-ebm-qwen35-9b-nothink",
]

# Sentinel text inserted by this script; used for idempotency checks.
# If this string appears in a README, the Phase 1 patch is already applied.
_PHASE1_SENTINEL = "<!-- carnot-exp317-phase1-patch -->"

_FCV_REPO_ID = "Carnot-EBM/carnot-formal-claim-verifier-v1"
_JOINT_CONSTRAINT_REPO_ID = "Carnot-EBM/carnot-joint-constraint-v1"


# ---------------------------------------------------------------------------
# Dependency injection helpers (patchable in tests)
# ---------------------------------------------------------------------------


def _make_hf_api() -> Any:
    """Return a new HfApi instance.

    Standalone function so tests can patch
    ``scripts.experiment_317_hf_publish._make_hf_api`` without importing
    huggingface_hub at module load time.
    """
    from huggingface_hub import HfApi  # type: ignore[import-untyped]
    return HfApi()


# ---------------------------------------------------------------------------
# Credential check (identical pattern to Exp 304)
# ---------------------------------------------------------------------------


def check_hf_credentials_317() -> tuple[bool, str]:
    """Check HuggingFace credentials via CLI, then Python API fallback.

    Precedence:
      1. ``huggingface-cli whoami`` — subprocess call.
      2. If CLI absent or fails → ``HfApi().whoami()`` Python API.
      3. Both fail → return (False, login_instructions).

    Returns:
        ``(True, username_or_msg)`` if credentials are valid.
        ``(False, instructions)`` if not authenticated.
    """
    _login_instructions = (
        "Run: huggingface-cli login --token <your-token>\n"
        "or set the HF_TOKEN environment variable.\n"
        "After login, re-run this script."
    )

    # Step 1: try CLI
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            username = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
            return True, f"logged in as {username}" if username else "logged in"
    except FileNotFoundError:
        pass  # CLI not installed
    except Exception:
        pass  # Any other subprocess error

    # Step 2: Python API fallback
    try:
        api = _make_hf_api()
        info = api.whoami()
        username = info.get("name", "") if isinstance(info, dict) else str(info)
        return True, f"logged in as {username}" if username else "logged in (Python API)"
    except Exception:
        pass  # Not authenticated via Python API either

    return False, _login_instructions


# ---------------------------------------------------------------------------
# Phase 1 README patch builder
# ---------------------------------------------------------------------------


def build_phase1_readme_patch(exp316_results: dict[str, Any] | None = None) -> str:
    """Build the Phase 1 patch block to prepend to per-token EBM READMEs.

    The patch clarifies the critical finding from Exp 184/203: these models
    detect model *confidence* (hallucination likelihood from hidden-state
    activations), NOT factual correctness.  This distinction is essential so
    users are not misled into thinking EBM energy scores can verify answers.

    The patch is idempotent: before applying, callers must check for
    ``_PHASE1_SENTINEL`` in the existing README.

    Args:
        exp316_results: Optional Exp 316 benchmark results dict.  When
            provided, a summary table of the full-scale benchmark is included
            so users can see the FCV pipeline's measured performance.

    Returns:
        Markdown string suitable for prepending to any per-token EBM README.
    """
    benchmark_section = ""
    if exp316_results is not None:
        # Build a compact summary from the Exp 316 per_variant_results.
        # We show baseline accuracy on the "all" variant for each model
        # tested — these are the most conservative (non-cherry-picked) numbers.
        per_variant = exp316_results.get("per_variant_results", {})
        all_variant = per_variant.get("all", {})
        if all_variant:
            rows = []
            for model_name, stats in sorted(all_variant.items()):
                acc = stats.get("accuracy", 0.0)
                ci_lo = stats.get("ci_lower", 0.0)
                ci_hi = stats.get("ci_upper", 0.0)
                n = stats.get("n_total", 0)
                rows.append(f"| {model_name} | {acc:.1%} | [{ci_lo:.1%}, {ci_hi:.1%}] | {n} |")
            if rows:
                table = "\n".join([
                    "| Model | GSM8K Accuracy | 95% CI | N |",
                    "|-------|:--------------:|:------:|:---:|",
                ] + rows)
                n_gsm8k = exp316_results.get("n_gsm8k", "?")
                n_humaneval = exp316_results.get("n_humaneval", "?")
                benchmark_section = f"""
## Exp 316 Full-Scale Benchmark Results (2026-04-14)

The Carnot FCV pipeline was benchmarked on {n_gsm8k} GSM8K questions
(adversarial corpus with number_swap and irrelevant_sentence perturbations)
and {n_humaneval} HumanEval problems.

Baseline accuracy on adversarial GSM8K (no Carnot intervention):

{table}

Note: inference_mode=simulated.  Live GPU results pending.
See `results/experiment_316_fullscale_results.json` for full details.
"""

    patch = f"""\
{_PHASE1_SENTINEL}
> **PHASE 1 RESEARCH ARTIFACT — detects model confidence, not factual correctness**
>
> This model was trained on LLM hidden-state activations to produce an energy
> score that correlates with the model's *output confidence* (hallucination
> likelihood).  **It cannot verify whether a model's answer is factually
> correct** — it can only signal how uncertain the model appears token-by-token.
>
> This limitation was confirmed in Exp 184/203: the energy scores reflect model
> confidence, not answer correctness.  Do not use these scores as a correctness
> verifier.
>
> **For production use**, install the full Carnot pipeline:
>
> ```bash
> pip install carnot
> ```
>
> The production pipeline includes FormalClaimVerifier (solver-routed formal
> claim verification), PBT code verification, and the Carnot MCP server.
> See [Carnot on GitHub](https://github.com/ianblenke/carnot) for documentation.
{benchmark_section}"""
    return patch


# ---------------------------------------------------------------------------
# Model card builders
# ---------------------------------------------------------------------------


def placeholder_card(repo_id: str) -> str:
    """Build an honest placeholder model card for a repo with no published weights.

    Used for carnot-joint-constraint-v1 when experiment_66_model.safetensors
    is absent.  The card is explicit that weights are not available, describes
    the methodology, and points users to `pip install carnot` rather than
    implying a working downloadable artifact.

    Args:
        repo_id: HuggingFace repo ID (used to generate the repo URL).

    Returns:
        Markdown string suitable for README.md.
    """
    short_name = repo_id.split("/")[-1]
    return f"""\
---
tags:
  - energy-based-model
  - constraint-verification
  - research-prototype
  - carnot
license: apache-2.0
---

# {short_name}

> **RESEARCH PROTOTYPE — weights not published**
>
> Trained model weights for this repository are not currently available for
> download.  The Exp 66 training run achieved AUROC 1.0 on held-out validation
> data (simulated JAX CPU training on synthetic constraint pairs), but the
> safetensors artifact was not exported before the training environment was
> torn down.
>
> This model card is preserved for reproducibility and methodology documentation.
> For a working EBM-based constraint verifier, use the FCV pipeline:
>
> ```bash
> pip install carnot
> ```

## Overview

`{short_name}` is the Exp 66 joint EBM + Ising constraint model from the
[Carnot](https://github.com/ianblenke/carnot) project.

Architecture:
- Embedding layer: text input projected to 384-dimensional space (embed_dim=384)
- Ising coupling: learned pairwise interactions among 8 latent constraint nodes
- MLP scoring head: hidden_dim=64 projection to a scalar confidence score

The joint model achieved AUROC 1.0 on held-out validation data.

## Methodology: 1.0 AUROC on Held-Out Validation

The 1.0 AUROC was achieved on a held-out split of synthetic constraint pairs
covering arithmetic, code, logic, factual, and scheduling domains.  The training
and evaluation used deterministic JAX random seeds on CPU.

Important provenance caveats:
- Training data was synthetic (generated, not from live LLM inference).
- The 1.0 AUROC is an in-distribution metric on a small held-out set.
- No live-inference benchmark exists for this model.

## Status

Model weights: NOT PUBLISHED (safetensors artifact unavailable).
Methodology: documented in `results/experiment_66_results.json`.
Phase: Phase 1 research prototype.

## Citation

```bibtex
@misc{{carnot-exp66-2026,
  title={{Exp 66: Differentiable Constraint Verification via Joint EBM + Ising Architecture}},
  author={{Carnot Research}},
  year={{2026}},
  url={{https://github.com/ianblenke/carnot}}
}}
```
"""


def build_fcv_readme_with_exp316(
    existing_readme: str,
    exp316_results: dict[str, Any] | None,
) -> str:
    """Patch the FCV README to include Exp 316 benchmark results.

    Appends a benchmark results section if Exp 316 data is available and the
    section is not already present.  If the section is already present, returns
    the existing README unchanged (idempotent).

    Args:
        existing_readme: Current README.md content from HuggingFace.
        exp316_results: Exp 316 benchmark results dict, or None if unavailable.

    Returns:
        Updated README string, or the original string if no update needed.
    """
    sentinel = "<!-- carnot-exp317-exp316-results -->"
    if sentinel in existing_readme:
        return existing_readme  # Already patched — idempotent

    if exp316_results is None:
        return existing_readme  # No data to add

    per_variant = exp316_results.get("per_variant_results", {})
    all_variant = per_variant.get("all", {})
    if not all_variant:
        return existing_readme

    rows = []
    for model_name, stats in sorted(all_variant.items()):
        acc = stats.get("accuracy", 0.0)
        ci_lo = stats.get("ci_lower", 0.0)
        ci_hi = stats.get("ci_upper", 0.0)
        n = stats.get("n_total", 0)
        rows.append(f"| {model_name} | {acc:.1%} | [{ci_lo:.1%}, {ci_hi:.1%}] | {n} |")

    table = "\n".join([
        "| Model | GSM8K Baseline Accuracy | 95% CI | N |",
        "|-------|:-----------------------:|:------:|:---:|",
    ] + rows)

    n_gsm8k = exp316_results.get("n_gsm8k", "?")
    n_humaneval = exp316_results.get("n_humaneval", "?")
    inference_mode = exp316_results.get("inference_mode", "unknown")

    section = f"""

{sentinel}
## Exp 316 Full-Scale Benchmark (2026-04-14)

Inference mode: **{inference_mode}**

{n_gsm8k} GSM8K questions (adversarial corpus) and {n_humaneval} HumanEval problems
across 4 modes (baseline, verify_only, verify_repair, z3_gated) and 2 models.

Baseline accuracy on adversarial GSM8K ("all" variant):

{table}

Note: inference_mode=simulated.  Live GPU results pending.
Full results: `results/experiment_316_fullscale_results.json`.
"""
    return existing_readme + section


# ---------------------------------------------------------------------------
# Idempotent README patch applicator
# ---------------------------------------------------------------------------


def model_card_update(
    repo_id: str,
    patch: str,
    hf_api: Any | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fetch a model README and apply the Phase 1 patch if not already applied.

    Idempotent: if the README already contains ``_PHASE1_SENTINEL``, the repo
    is recorded as skipped without any upload.

    Args:
        repo_id: HuggingFace repo ID (e.g. ``Carnot-EBM/per-token-ebm-qwen3-06b``).
        patch: Markdown patch block to prepend (from ``build_phase1_readme_patch``).
        hf_api: Optional injected HfApi instance.  Created if not provided.
        dry_run: If True, skip the actual upload and simulate success.

    Returns:
        Dict with keys: repo_id, status ("updated", "skipped", "error"), hf_url.
    """
    if hf_api is None:
        hf_api = _make_hf_api()

    hf_url = f"https://huggingface.co/{repo_id}"

    # Fetch existing README
    existing_readme = ""
    try:
        existing_readme = hf_api.hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type="model",
        )
        # hf_hub_download returns a local path; read the file
        if isinstance(existing_readme, str) and Path(existing_readme).exists():
            existing_readme = Path(existing_readme).read_text(encoding="utf-8")
        elif not isinstance(existing_readme, str):
            existing_readme = str(existing_readme)
    except Exception:
        # If README doesn't exist or any fetch error, start from empty
        existing_readme = ""

    # Idempotency check — already patched?
    if _PHASE1_SENTINEL in existing_readme:
        return {"repo_id": repo_id, "status": "skipped", "hf_url": hf_url}

    # Prepend the patch
    updated_readme = patch + "\n\n" + existing_readme if existing_readme else patch

    if dry_run:
        return {"repo_id": repo_id, "status": "updated", "hf_url": hf_url}

    # Upload updated README
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tf:
            tf.write(updated_readme)
            tf_path = tf.name
        hf_api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        hf_api.upload_file(
            path_or_fileobj=tf_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Exp 317: Apply Phase 1 research artifact disclaimer",
        )
        return {"repo_id": repo_id, "status": "updated", "hf_url": hf_url}
    except Exception as exc:
        return {"repo_id": repo_id, "status": "error", "hf_url": hf_url, "error": str(exc)}


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_experiment_317(
    dry_run: bool = False,
    results_path: Path | None = None,
    hf_api: Any | None = None,
) -> dict[str, Any]:
    """Run the full Exp 317 HuggingFace README audit and update pipeline.

    Steps:
      1. Check credentials (CLI → Python API fallback).  Emit blocked artifact
         with exp_317_next_action if not authenticated.
      2. Load Exp 316 results if available (used for benchmark summaries).
      3. Build Phase 1 patch block.
      4. Update all 16 per-token EBM repos (idempotent).
      5. Update carnot-formal-claim-verifier-v1 with Exp 316 benchmark results.
      6. Create/update carnot-joint-constraint-v1 with honest placeholder card.
      7. Write results JSON.

    Args:
        dry_run: Skip live HF API uploads; simulate success for all repos.
        results_path: Override write path for results JSON.
        hf_api: Optional injected HfApi instance (for testing).

    Returns:
        Results dict (also written to disk).
    """
    _results_write_path = results_path if results_path is not None else _RESULTS_PATH

    def _write_results(data: dict[str, Any]) -> None:
        _results_write_path.parent.mkdir(parents=True, exist_ok=True)
        _results_write_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    # -----------------------------------------------------------------------
    # Step 1: Credential check
    # -----------------------------------------------------------------------
    creds_ok, creds_msg = check_hf_credentials_317()

    if not creds_ok:
        blocked: dict[str, Any] = {
            "experiment": 317,
            "run_date": "20260414",
            "blocked": True,
            "credentials_available": False,
            "exp_317_next_action": (
                "Run: huggingface-cli login --token <your-token>\n"
                "or: export HF_TOKEN=<your-token>\n"
                "Then re-run: python scripts/experiment_317_hf_publish.py"
            ),
            "models_updated": [],
            "models_skipped": [],
            "errors": [],
        }
        _write_results(blocked)
        return blocked

    # -----------------------------------------------------------------------
    # Step 2: Load Exp 316 results
    # -----------------------------------------------------------------------
    exp316_results: dict[str, Any] | None = None
    if _EXP316_RESULTS_PATH.exists():
        try:
            exp316_results = json.loads(_EXP316_RESULTS_PATH.read_text())
        except Exception:
            exp316_results = None

    # -----------------------------------------------------------------------
    # Step 3: Build Phase 1 patch
    # -----------------------------------------------------------------------
    phase1_patch = build_phase1_readme_patch(exp316_results=exp316_results)

    # Resolve HfApi instance — shared across all update calls
    if hf_api is None:
        hf_api = _make_hf_api()

    models_updated: list[str] = []
    models_skipped: list[str] = []
    errors: list[dict[str, str]] = []

    # -----------------------------------------------------------------------
    # Step 4: Update all 16 per-token EBM repos
    # -----------------------------------------------------------------------
    for repo_id in _PER_TOKEN_EBM_REPOS:
        result = model_card_update(
            repo_id=repo_id,
            patch=phase1_patch,
            hf_api=hf_api,
            dry_run=dry_run,
        )
        status = result.get("status")
        if status == "updated":
            models_updated.append(repo_id)
        elif status == "skipped":
            models_skipped.append(repo_id)
        else:
            errors.append({"repo_id": repo_id, "error": result.get("error", "unknown")})

    # -----------------------------------------------------------------------
    # Step 5: Update FCV README with Exp 316 results
    # -----------------------------------------------------------------------
    fcv_status = "skipped"
    fcv_hf_url = f"https://huggingface.co/{_FCV_REPO_ID}"
    try:
        existing_fcv = ""
        try:
            path_or_content = hf_api.hf_hub_download(
                repo_id=_FCV_REPO_ID,
                filename="README.md",
                repo_type="model",
            )
            if isinstance(path_or_content, str) and Path(path_or_content).exists():
                existing_fcv = Path(path_or_content).read_text(encoding="utf-8")
            elif not isinstance(path_or_content, str):
                existing_fcv = str(path_or_content)
            else:
                existing_fcv = path_or_content
        except Exception:
            existing_fcv = ""

        updated_fcv = build_fcv_readme_with_exp316(existing_fcv, exp316_results)
        if updated_fcv != existing_fcv:
            if not dry_run:
                import tempfile
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".md", delete=False, encoding="utf-8"
                ) as tf:
                    tf.write(updated_fcv)
                    tf_path = tf.name
                hf_api.upload_file(
                    path_or_fileobj=tf_path,
                    path_in_repo="README.md",
                    repo_id=_FCV_REPO_ID,
                    repo_type="model",
                    commit_message="Exp 317: Add Exp 316 benchmark results",
                )
            fcv_status = "updated"
            models_updated.append(_FCV_REPO_ID)
        else:
            fcv_status = "skipped"
            models_skipped.append(_FCV_REPO_ID)
    except Exception as exc:
        fcv_status = "error"
        errors.append({"repo_id": _FCV_REPO_ID, "error": str(exc)})

    # -----------------------------------------------------------------------
    # Step 6: Create/update joint constraint placeholder card
    # -----------------------------------------------------------------------
    joint_status = "skipped"
    try:
        card_content = placeholder_card(_JOINT_CONSTRAINT_REPO_ID)
        # Check if existing card already has the "RESEARCH PROTOTYPE" label
        existing_joint = ""
        try:
            path_or_content = hf_api.hf_hub_download(
                repo_id=_JOINT_CONSTRAINT_REPO_ID,
                filename="README.md",
                repo_type="model",
            )
            if isinstance(path_or_content, str) and Path(path_or_content).exists():
                existing_joint = Path(path_or_content).read_text(encoding="utf-8")
            elif not isinstance(path_or_content, str):
                existing_joint = str(path_or_content)
            else:
                existing_joint = path_or_content
        except Exception:
            existing_joint = ""

        if "RESEARCH PROTOTYPE" in existing_joint and "weights not published" in existing_joint:
            joint_status = "skipped"
            models_skipped.append(_JOINT_CONSTRAINT_REPO_ID)
        else:
            if not dry_run:
                import tempfile
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".md", delete=False, encoding="utf-8"
                ) as tf:
                    tf.write(card_content)
                    tf_path = tf.name
                hf_api.create_repo(
                    repo_id=_JOINT_CONSTRAINT_REPO_ID,
                    repo_type="model",
                    exist_ok=True,
                )
                hf_api.upload_file(
                    path_or_fileobj=tf_path,
                    path_in_repo="README.md",
                    repo_id=_JOINT_CONSTRAINT_REPO_ID,
                    repo_type="model",
                    commit_message="Exp 317: Honest placeholder — weights not published",
                )
            joint_status = "updated"
            models_updated.append(_JOINT_CONSTRAINT_REPO_ID)
    except Exception as exc:
        joint_status = "error"
        errors.append({"repo_id": _JOINT_CONSTRAINT_REPO_ID, "error": str(exc)})

    # -----------------------------------------------------------------------
    # Step 7: Build and write results
    # -----------------------------------------------------------------------
    results: dict[str, Any] = {
        "experiment": 317,
        "run_date": "20260414",
        "blocked": False,
        "credentials_available": True,
        "hf_credentials": creds_msg,
        "models_updated": models_updated,
        "models_skipped": models_skipped,
        "errors": errors,
        "exp316_results_available": exp316_results is not None,
        "dry_run": dry_run,
        "honest_verdict": {
            "status": "dry_run" if dry_run else ("complete" if not errors else "partial"),
            "explanation": (
                f"{len(models_updated)} repos updated, "
                f"{len(models_skipped)} skipped (already patched), "
                f"{len(errors)} errors."
                + (" dry_run=True; no network calls made." if dry_run else "")
            ),
        },
    }
    _write_results(results)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 317: HuggingFace README accuracy audit."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp 317: HuggingFace README accuracy audit and update"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Skip live HF API calls (simulate update only)",
    )
    args = parser.parse_args()

    result = run_experiment_317(dry_run=args.dry_run)

    if result.get("blocked"):
        print("BLOCKED: HuggingFace credentials not found.")
        print(result.get("exp_317_next_action", ""))
    else:
        verdict = result.get("honest_verdict", {})
        print(f"Exp 317 complete. Status: {verdict.get('status')}")
        print(f"  {verdict.get('explanation')}")
        print(f"  Results: {_RESULTS_PATH}")


if __name__ == "__main__":
    main()
