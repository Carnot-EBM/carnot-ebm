#!/usr/bin/env python3
"""Experiment 413: EnvironmentAutoFix + GPU preflight v3.

**Purpose:**
    Re-run GPU preflight after applying EnvironmentAutoFix to test whether
    self-injecting ``CARNOT_FORCE_LIVE=1`` (when GPU hardware is present but the
    var is absent from the conductor subprocess) resolves RETRO-022.

**Why EnvironmentAutoFix is called before ExperimentTemplate:**
    ExperimentTemplate reads ``CARNOT_FORCE_LIVE`` during ``setup_gpu()``.  If we
    inject the var after construction, ``setup_gpu()`` will still see the stale env.
    By calling ``apply_env_autofix()`` first, we guarantee the var is present for
    all downstream code including ExperimentTemplate.

**Honest verdict semantics:**
    - ``'gpu_confirmed_live'``           — GPU detected AND CARNOT_FORCE_LIVE='1'
    - ``'gpu_detected_env_was_correct'`` — GPU detected AND var was already set
    - ``'gpu_not_detected'``             — no GPU hardware
    - ``'auto_fix_applied'``             — var was absent but auto-fix injected it

**Output:** ``results/experiment_413_env_autofix.json``

This is a CPU-safe experiment: it always completes and produces a result JSON
regardless of GPU availability.

Spec: REQ-INFRA-021, REQ-INFRA-022,
      SCENARIO-INFRA-025, SCENARIO-INFRA-026, SCENARIO-INFRA-027 (Exp 413)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# STEP 0: Apply EnvironmentAutoFix BEFORE any other import that reads
#         CARNOT_FORCE_LIVE from os.environ.  This is the whole point of the
#         module — calling it here, at the very top of main(), ensures every
#         downstream code path (ExperimentTemplate, LiveGPUGate, etc.) sees
#         the correct env state.
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import (
    apply_env_autofix,
    build_env_autofix_artifact,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure project scripts directory is importable
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.deliverable_validator import DeliverableContentValidator  # noqa: E402
from carnot.pipeline.gpu_preflight import run_gpu_preflight  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

_DELIVERABLE = "results/experiment_413_env_autofix.json"
_PRIOR_PREFLIGHT_PATH = _REPO_ROOT / "results" / "experiment_404_preflight_v2.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_prior_preflight(path: Path) -> dict:
    """Load prior preflight JSON; return empty dict with note if missing/corrupt."""
    if not path.exists():
        return {"note": "prior_preflight_file_missing", "path": str(path)}
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text)
    except (json.JSONDecodeError, OSError) as exc:
        return {"note": f"prior_preflight_file_unreadable: {exc}", "path": str(path)}


def _count_corrupt_files(project_root: Path) -> int:
    """Return count of RETRO-023 known corrupt files still present."""
    audit = DeliverableContentValidator.audit_known_corrupt_files(str(project_root))
    return sum(1 for v in audit.values() if v == "corrupt_json")


def _print_action_required(honest_verdict: str) -> None:
    """Print cloud GPU action required message when verdict is non-live."""
    print()
    print("=" * 72)
    print("ACTION REQUIRED — GPU not confirmed live")
    print(f"honest_verdict = {honest_verdict!r}")
    print()
    print("To run live GPU experiments, provision a cloud GPU:")
    print()
    print("  Option 1: Lambda Labs")
    print("    lambdalabs instance create --instance-type gpu_1x_a100 \\")
    print("      --region us-west-2 --quantity 1")
    print()
    print("  Option 2: vast.ai")
    print("    vastai search offers 'gpu_name=A100' --storage 50")
    print("    vastai create instance <id> --image pytorch/pytorch:2.3.0-cuda12.1")
    print()
    print("  Option 3: RunPod")
    print("    runpodctl create pod --gpuType NVIDIA_A100_80GB")
    print()
    print("Then re-run with CARNOT_FORCE_LIVE=1 in the environment.")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> dict:
    """Run EnvironmentAutoFix + GPU preflight v3 and write result artifact.

    Returns
    -------
    dict
        The written artifact.
    """
    # --- Step A: Apply env autofix (BEFORE ExperimentTemplate) ---
    autofix_result = apply_env_autofix()

    # --- Step B: ExperimentTemplate setup ---
    tmpl = ExperimentTemplate(
        413,
        "EnvironmentAutoFix + GPU preflight v3",
        _DELIVERABLE,
        repo_root=_REPO_ROOT,
    )
    tmpl.setup()

    # --- Step C: Re-run GPU preflight with updated env ---
    preflight = run_gpu_preflight(_REPO_ROOT, model_ids=[])

    # --- Step D: Load prior preflight for comparison ---
    prior_preflight = _load_prior_preflight(_PRIOR_PREFLIGHT_PATH)

    # --- Step E: Count remaining corrupt files ---
    n_corrupt_files_remaining = _count_corrupt_files(_REPO_ROOT)

    # --- Step F: Build artifact ---
    env_autofix_art = build_env_autofix_artifact(autofix_result, prior_preflight)
    honest_verdict = env_autofix_art["honest_verdict"]
    retro_022_resolved = env_autofix_art["retro_022_resolved"]

    artifact = tmpl.build_result(
        {
            "schema": "carnot.env_autofix.v1",
            "honest_verdict": honest_verdict,
            "retro_022_resolved": retro_022_resolved,
            "auto_fix_applied": autofix_result.auto_fix_applied,
            "gpu_detected": autofix_result.gpu_detected,
            "carnot_force_live_was_set": autofix_result.carnot_force_live_was_set,
            "final_env_value": autofix_result.final_env_value,
            "preflight_v3": {
                "honest_verdict": preflight.honest_verdict,
                "env_var_set": preflight.env_var_set,
                "subprocess_inherits_env": preflight.subprocess_inherits_env,
                "is_live_capable": preflight.is_live_capable,
                "smoke_test_passed": preflight.smoke_test_passed,
                "retro_019_resolved": preflight.retro_019_resolved,
            },
            "prior_preflight_verdict": prior_preflight.get("honest_verdict", "unknown"),
            "n_corrupt_files_remaining": n_corrupt_files_remaining,
        },
        status="success",
    )

    # --- Step G: Write artifact ---
    out_path = _REPO_ROOT / _DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    # --- Step H: Print summary ---
    print(f"\nExp 413 complete — honest_verdict={honest_verdict!r}")
    print(f"  auto_fix_applied     = {autofix_result.auto_fix_applied}")
    print(f"  gpu_detected         = {autofix_result.gpu_detected}")
    print(f"  retro_022_resolved   = {retro_022_resolved}")
    print(f"  n_corrupt_remaining  = {n_corrupt_files_remaining}")
    print(f"  preflight_v3         = {preflight.honest_verdict!r}")
    print(f"  artifact             → {_DELIVERABLE}")

    # --- Step I: ACTION REQUIRED message if not live ---
    if honest_verdict not in ("gpu_confirmed_live", "auto_fix_applied"):
        _print_action_required(honest_verdict)

    return artifact


if __name__ == "__main__":
    main()
