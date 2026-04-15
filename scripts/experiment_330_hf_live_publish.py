"""Experiment 330: Live HuggingFace publish with Exp 328 live-GPU benchmarks embedded.

Wraps the Exp 317 HuggingFace README publish pipeline and additionally embeds
live-GPU benchmark results from Exp 328 into all 16 per-token activation EBM
model READMEs on HuggingFace.

Why this matters:
  Exp 317 was blocked by missing HF credentials (no HF_TOKEN).  Now that
  credentials are available, this script executes the live publish.  It also
  upgrades the embedded benchmark numbers from Exp 316 (simulated) to Exp 328
  (live_gpu, measured on two RTX 3090 GPUs on 2026-04-15).

Live benchmark provenance (Exp 328 / first_live_run_evidence):
  - Qwen3.5-0.8B: 27.5% baseline accuracy on adversarial GSM8K (all variant)
  - Gemma4-E4B-it: 26.3% baseline accuracy on adversarial GSM8K (all variant)
  These numbers supersede the Exp 316 simulated values (34.0% / 30.0%).

Idempotency:
  If the Phase 1 sentinel (<!-- carnot-exp317-phase1-patch -->) is already
  present in a README, that repo is skipped.  The n_idempotency_checks_passed
  counter records how many already-updated repos were confirmed sentinel-present.

Spec: REQ-PUBLISH-004, SCENARIO-PUBLISH-007, SCENARIO-PUBLISH-008
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
_RESULTS_PATH = _REPO_ROOT / "results" / "experiment_330_hf_publish_results.json"
_EXP328_RESULTS_PATH = _REPO_ROOT / "results" / "experiment_328_live_fullscale_results.json"
_EXP316_RESULTS_PATH = _REPO_ROOT / "results" / "experiment_316_fullscale_results.json"

# The sentinel used by Exp 317 for idempotency checks.
_PHASE1_SENTINEL = "<!-- carnot-exp317-phase1-patch -->"


# ---------------------------------------------------------------------------
# Dependency injection helper
# ---------------------------------------------------------------------------


def _make_hf_api_330() -> Any:
    """Return a new HfApi instance.

    Standalone function so tests can patch
    ``scripts.experiment_330_hf_live_publish._make_hf_api_330`` without
    importing huggingface_hub at module load time.
    """
    from huggingface_hub import HfApi  # type: ignore[import-untyped]
    return HfApi()


# ---------------------------------------------------------------------------
# Public helpers (tested independently)
# ---------------------------------------------------------------------------


def load_publish_results(path: Path) -> dict[str, Any]:
    """Load and validate a publish results JSON file.

    Validates that the file exists, is valid JSON, and contains the required
    'experiment' and 'status' keys.  These are the minimum fields needed to
    determine whether a publish run succeeded.

    Args:
        path: Filesystem path to the results JSON file.

    Returns:
        Parsed results dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or missing required keys.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse error in {path}: {exc}") from exc
    if "experiment" not in data:
        raise ValueError(f"Missing required key 'experiment' in {path}")
    if "status" not in data:
        raise ValueError(f"Missing required key 'status' in {path}")
    return data


def validate_live_publish(result: dict[str, Any]) -> None:
    """Raise ValueError if a publish result is not successful.

    Used to gate idempotency verification: if the publish did not succeed
    (e.g. blocked by missing credentials or partial errors), we should not
    claim the operation is done.

    Args:
        result: Publish result dict (must contain 'status').

    Raises:
        ValueError: If status != 'success'.
    """
    status = result.get("status", "")
    if status != "success":
        raise ValueError(
            f"Publish result status is '{status}', expected 'success'. "
            f"Cannot validate live publish."
        )


def adapt_exp328_to_per_variant(exp328: dict[str, Any]) -> dict[str, Any] | None:
    """Convert Exp 328 live-GPU results to per_variant_results format.

    Exp 328's first_live_run_evidence keys follow the naming pattern:
      <Model>_baseline_all_accuracy  →  float
      <Model>_baseline_all_ci        →  str like "[0.234, 0.321]"

    These are the most authoritative live numbers because they come directly
    from the first benchmark run before simulated re-runs overwrote the JSON.

    We adapt them to the per_variant_results["all"] dict structure that
    build_phase1_readme_patch() expects, so the live numbers appear in the
    Phase 1 disclaimer block on HuggingFace.

    Args:
        exp328: Loaded Exp 328 results dict.  May be empty if file was absent.

    Returns:
        Adapted results dict with 'per_variant_results' key, or None if the
        required evidence fields are absent.
    """
    evidence = exp328.get("first_live_run_evidence", {})
    if not evidence:
        return None

    # Collect model accuracy entries from first_live_run_evidence.
    # Keys follow: "<model_name>_baseline_all_accuracy" and
    # "<model_name>_baseline_all_ci".
    all_variant: dict[str, dict[str, Any]] = {}
    # Find all _baseline_all_accuracy keys
    for key, val in evidence.items():
        if key.endswith("_baseline_all_accuracy") and isinstance(val, float):
            model_name = key[: -len("_baseline_all_accuracy")]
            ci_key = f"{model_name}_baseline_all_ci"
            ci_str = evidence.get(ci_key, "[0.0, 1.0]")
            # Parse "[lo, hi]" string
            ci_lo, ci_hi = 0.0, 1.0
            try:
                stripped = ci_str.strip("[]").split(",")
                ci_lo = float(stripped[0].strip())
                ci_hi = float(stripped[1].strip())
            except Exception:
                pass
            all_variant[model_name] = {
                "accuracy": val,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "n_total": exp328.get("benchmark_n_gsm8k", 0),
            }

    if not all_variant:
        return None

    return {
        "per_variant_results": {"all": all_variant},
        "n_gsm8k": exp328.get("benchmark_n_gsm8k", 0),
        "n_humaneval": exp328.get("benchmark_n_humaneval", 0),
        "inference_mode": "live_gpu",
        "source_experiment": 328,
        "run_date": exp328.get("run_date", "20260415"),
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_experiment_330(
    dry_run: bool = False,
    results_path: Path | None = None,
    exp328_results_path: Path | None = None,
    hf_api: Any | None = None,
) -> dict[str, Any]:
    """Run the Exp 330 live HuggingFace publish with Exp 328 benchmark embedding.

    Steps:
      1. Check HF credentials.  Emit blocked artifact if not authenticated.
      2. Load Exp 328 live-GPU results; adapt to per_variant_results format.
         Fall back to Exp 316 simulated results if Exp 328 is absent.
      3. Delegate to run_experiment_317 (live), passing adapted benchmark data.
      4. Count idempotency checks passed (sentinels verified in updated repos).
      5. Build and write exp330 wrapper artifact.

    Args:
        dry_run: If True, skip live HF API uploads (simulate success).
        results_path: Override write path.  Defaults to
            results/experiment_330_hf_publish_results.json.
        exp328_results_path: Override path to Exp 328 results.
        hf_api: Optional injected HfApi instance (for testing).

    Returns:
        Results dict (also written to disk).
    """
    from scripts.experiment_317_hf_publish import (
        _PER_TOKEN_EBM_REPOS,
        check_hf_credentials_317,
        run_experiment_317,
    )

    _write_path = results_path if results_path is not None else _RESULTS_PATH
    _exp328_path = exp328_results_path if exp328_results_path is not None else _EXP328_RESULTS_PATH

    def _write(data: dict[str, Any]) -> None:
        Path(_write_path).parent.mkdir(parents=True, exist_ok=True)
        Path(_write_path).write_text(json.dumps(data, indent=2, sort_keys=True))

    # -----------------------------------------------------------------------
    # Step 1: Credential check
    # -----------------------------------------------------------------------
    creds_ok, creds_msg = check_hf_credentials_317()

    if not creds_ok:
        blocked: dict[str, Any] = {
            "experiment": 330,
            "schema": "carnot.hf_publish.v1",
            "run_date": "20260415",
            "status": "blocked",
            "next_action": (
                "huggingface-cli login --token <your-token>\n"
                "or: export HF_TOKEN=<your-token>\n"
                "Then re-run: python scripts/experiment_330_hf_live_publish.py"
            ),
            "n_models_updated": 0,
            "n_models_skipped": 0,
            "fcv_updated": False,
            "joint_placeholder_created": False,
            "live_benchmark_embedded": False,
            "exp328_baseline_accuracy": {},
            "n_idempotency_checks_passed": 0,
        }
        _write(blocked)
        return blocked

    # -----------------------------------------------------------------------
    # Step 2: Load Exp 328 live-GPU results (fall back to Exp 316 if absent)
    # -----------------------------------------------------------------------
    adapted_benchmark: dict[str, Any] | None = None
    live_benchmark_embedded = False
    exp328_baseline_accuracy: dict[str, float] = {}

    if Path(_exp328_path).exists():
        try:
            exp328_raw = json.loads(Path(_exp328_path).read_text(encoding="utf-8"))
            adapted_benchmark = adapt_exp328_to_per_variant(exp328_raw)
            if adapted_benchmark is not None:
                live_benchmark_embedded = True
                # Extract per-model accuracy for the artifact
                all_variant = adapted_benchmark.get("per_variant_results", {}).get("all", {})
                exp328_baseline_accuracy = {
                    model: stats.get("accuracy", 0.0)
                    for model, stats in all_variant.items()
                }
        except Exception:
            adapted_benchmark = None

    # Fall back to Exp 316 simulated if Exp 328 not available
    if adapted_benchmark is None and _EXP316_RESULTS_PATH.exists():
        try:
            adapted_benchmark = json.loads(_EXP316_RESULTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            adapted_benchmark = None

    # -----------------------------------------------------------------------
    # Step 3: Run Exp 317 publish pipeline (live or dry-run)
    # -----------------------------------------------------------------------
    # We write a temporary merged results path so exp317 doesn't clobber
    # the exp316 path.  exp317 writes its own results; we only need the
    # return value.
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=str(_write_path.parent)
        if hasattr(_write_path, "parent") else "/tmp"
    ) as tf:
        tf317_path = Path(tf.name)

    # Patch exp316 results path temporarily so build_phase1_readme_patch
    # picks up Exp 328 adapted data instead.
    # We do this by passing the adapted data via a temp file.
    if adapted_benchmark is not None:
        tf317_path.write_text(json.dumps(adapted_benchmark))

    # Resolve HfApi for exp317 calls
    if hf_api is None:
        hf_api = _make_hf_api_330()

    # We need to override exp317's _EXP316_RESULTS_PATH.  The cleanest way
    # is to patch the module constant temporarily.
    import scripts.experiment_317_hf_publish as _exp317_mod

    _orig_316_path = _exp317_mod._EXP316_RESULTS_PATH
    if adapted_benchmark is not None:
        _exp317_mod._EXP316_RESULTS_PATH = tf317_path  # type: ignore[assignment]

    try:
        exp317_result = run_experiment_317(
            dry_run=dry_run,
            results_path=Path(_write_path).parent / "experiment_317_hf_publish.json",
            hf_api=hf_api,
        )
    finally:
        _exp317_mod._EXP316_RESULTS_PATH = _orig_316_path  # type: ignore[assignment]
        try:
            tf317_path.unlink(missing_ok=True)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Step 4: Count idempotency checks passed
    # -----------------------------------------------------------------------
    # For each repo that exp317 recorded as "skipped" (already patched),
    # the sentinel was confirmed present — these are idempotency passes.
    # For repos that were updated, we trust the sentinel was just written.
    n_idempotency_checks_passed = len(exp317_result.get("models_skipped", []))
    models_updated: list[str] = exp317_result.get("models_updated", [])
    models_skipped: list[str] = exp317_result.get("models_skipped", [])
    errors: list[dict[str, str]] = exp317_result.get("errors", [])

    # Derive high-level flags from the exp317 model lists
    fcv_updated = "Carnot-EBM/carnot-formal-claim-verifier-v1" in models_updated
    joint_placeholder_created = (
        "Carnot-EBM/carnot-joint-constraint-v1" in models_updated
    )
    # n_models_updated counts per-token EBM repos only (excludes FCV / joint)
    per_token_updated = [r for r in models_updated if "per-token-ebm" in r]
    per_token_skipped = [r for r in models_skipped if "per-token-ebm" in r]
    n_models_updated = len(per_token_updated)
    n_models_skipped = len(per_token_skipped)

    # -----------------------------------------------------------------------
    # Step 5: Build and write exp330 artifact
    # -----------------------------------------------------------------------
    overall_status = "success"
    if errors:
        # Partial success: some repos errored
        overall_status = "partial"
    if exp317_result.get("blocked"):
        overall_status = "blocked"

    artifact: dict[str, Any] = {
        "experiment": 330,
        "schema": "carnot.hf_publish.v1",
        "run_date": "20260415",
        "status": overall_status,
        "n_models_updated": n_models_updated,
        "n_models_skipped": n_models_skipped,
        "fcv_updated": fcv_updated,
        "joint_placeholder_created": joint_placeholder_created,
        "live_benchmark_embedded": live_benchmark_embedded,
        "exp328_baseline_accuracy": exp328_baseline_accuracy,
        "n_idempotency_checks_passed": n_idempotency_checks_passed,
        "all_models_updated": models_updated,
        "all_models_skipped": models_skipped,
        "errors": errors,
        "dry_run": dry_run,
        "exp317_status": exp317_result.get("honest_verdict", {}).get("status", "unknown"),
    }
    _write(artifact)
    return artifact


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 330: Live HuggingFace publish with Exp 328 benchmark embedding."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp 330: Live HF publish with live-GPU benchmark embedding"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Skip live HF API calls (simulate update only)",
    )
    args = parser.parse_args()

    result = run_experiment_330(dry_run=args.dry_run)

    if result.get("status") == "blocked":
        print("BLOCKED: HuggingFace credentials not found.")
        print(result.get("next_action", ""))
    else:
        print(f"Exp 330 complete. Status: {result['status']}")
        print(f"  Models updated: {result['n_models_updated']}")
        print(f"  Models skipped (already patched): {result['n_models_skipped']}")
        print(f"  FCV README updated: {result['fcv_updated']}")
        print(f"  Joint constraint placeholder: {result['joint_placeholder_created']}")
        print(f"  Live benchmark embedded: {result['live_benchmark_embedded']}")
        print(f"  Idempotency checks passed: {result['n_idempotency_checks_passed']}")
        if result.get("errors"):
            print(f"  Errors: {len(result['errors'])}")
            for err in result["errors"]:
                print(f"    - {err.get('repo_id')}: {err.get('error')}")
        print(f"  Results: {_RESULTS_PATH}")


if __name__ == "__main__":
    main()
