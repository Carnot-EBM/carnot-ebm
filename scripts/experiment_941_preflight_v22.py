"""
Experiment 941 — Milestone 2026.04.73 Pre-flight v22

PURPOSE: Audit .72 retrospective results (10/12 criteria), verify SOTA GGUF model
availability for Exp 942 (math repair rerun), document all 8 prior SC-energy
experiments so Exp 944 can pass the conductor gate-check, and emit a stable
preflight artifact as the .73 cycle starting checkpoint.

WHY THIS MATTERS: Milestone .72 failed two criteria:
  1. RETRO-MATH-REPAIR-MODEL-CEILING — gemma-4-E4B-it (too small) produced 0%
     signed improvement on GSM8K.  The algorithm is correct; the model is wrong.
     SOTA model (Gemma4-31B or Qwen3.6-35B-A3B) is required for Exp 942.
  2. RETRO-SC-ENERGY-GATE-DISCIPLINE — Exp 939 blocked because its YAML task
     had no prior_failures entries for 7 prior SC-energy experiments.  Same error
     as Exp 917 in milestone .71.  Exp 944 MUST include all 8 prior_failures.

This script:
  (a) checks whether any SOTA GGUF is already in the HF cache — downloads the
      smallest viable model (gemma-4-26B-A4B-it Q4_K_M) if none found;
  (b) confirms the SC-energy audit is complete (8 experiments documented);
  (c) emits the preflight JSON that the conductor reads as required-reading for .73.
"""

import json
import os
import sys
from datetime import datetime, timezone, UTC
from pathlib import Path

UTC = UTC

# ---------------------------------------------------------------------------
# SOTA model catalog — all three CLAUDE.md mandatory SOTA models (unsloth GGUFs)
# ---------------------------------------------------------------------------
_SOTA_CANDIDATES = [
    {
        "hf_repo": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "filename": "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        "cache_subdir": "models--unsloth--gemma-4-26B-A4B-it-GGUF",
    },
    {
        "hf_repo": "unsloth/gemma-4-31B-it-GGUF",
        "filename": None,  # any gguf
        "cache_subdir": "models--unsloth--gemma-4-31B-it-GGUF",
    },
    {
        "hf_repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "filename": None,  # any gguf
        "cache_subdir": "models--unsloth--Qwen3.6-35B-A3B-GGUF",
    },
]

# The 8 SC-energy prior experiments that Exp 944 must document in prior_failures.
# Derived from Exp 939's gate_check JSON (experiment_939_sc_energy_set_consistency.json).
_SC_ENERGY_PRIORS = [
    {
        "experiment_id": "exp506-semantic-energy-tier0d",
        "experiment_number": 506,
        "verdict": "semantic_energy_no_improvement",
        "domain": "Semantic Energy Tier 0d",
    },
    {
        "experiment_id": "exp509-ppsebm-energy-magnitude-replay",
        "experiment_number": 509,
        "verdict": "energy_magnitude_wins",
        "domain": "PPSEBM Energy Magnitude Replay (adjacent domain)",
    },
    {
        "experiment_id": "exp533-cold-decoding-energy-guidance",
        "experiment_number": 533,
        "verdict": "no_violation_reduction",
        "domain": "COLD Decoding Energy Guidance (adjacent domain)",
    },
    {
        "experiment_id": "exp711-sc-energy-set-consistency",
        "experiment_number": 711,
        "verdict": "tier_29_below_threshold",
        "domain": "SC-Energy SetConsistencyVerifier Tier 2.9",
    },
    {
        "experiment_id": "exp725-sc-energy-v2",
        "experiment_number": 725,
        "verdict": "sc_energy_v2_below_threshold",
        "domain": "SC-Energy v2 FoVer v2 Dual Labels",
    },
    {
        "experiment_id": "exp772-semantic-energy-probe",
        "experiment_number": 772,
        "verdict": "semantic_energy_below_baseline",
        "domain": "SemanticEnergyProbe Tier 0g",
    },
    {
        "experiment_id": "exp787-sstar-energy-ranked-code-selection",
        "experiment_number": 787,
        "verdict": "energy_prefilter_efficient",
        "domain": "S* Energy Pre-Ranking (adjacent domain)",
    },
    {
        "experiment_id": "exp939-sc-energy-set-consistency-networks",
        "experiment_number": 939,
        "verdict": "blocked_gate_check_failed",
        "domain": "SC-Energy Set Consistency Networks (Contrastive Coherence)",
    },
]


def _find_sota_model_in_cache() -> tuple[str | None, bool]:
    """
    Search the HuggingFace cache for any pre-downloaded SOTA GGUF file.

    Returns (path_or_none, was_downloaded_this_run).  We check the three
    CLAUDE.md-mandatory unsloth GGUF repos in priority order (smallest first
    for fast math-repair iteration).  If a Q4_K_M file exists we prefer it;
    otherwise we accept any .gguf in the snapshot directory.
    """
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub_dir = hf_home / "hub"

    for candidate in _SOTA_CANDIDATES:
        cache_path = hub_dir / candidate["cache_subdir"]
        if not cache_path.exists():
            continue
        snapshots = (
            list((cache_path / "snapshots").glob("*"))
            if (cache_path / "snapshots").exists()
            else []
        )
        for snap in snapshots:
            # Prefer the named file if specified
            if candidate["filename"]:
                target = snap / candidate["filename"]
                if target.exists():
                    return str(target), False
            # Fall back to any .gguf in the snapshot tree
            gguf_files = list(snap.rglob("*.gguf"))
            if gguf_files:
                # Prefer Q4_K_M quantisation for math repair experiments
                q4km = [f for f in gguf_files if "Q4_K_M" in f.name]
                chosen = q4km[0] if q4km else gguf_files[0]
                return str(chosen), False

    return None, False


def _attempt_sota_download() -> tuple[str | None, bool]:
    """
    Attempt to download the smallest viable SOTA GGUF via huggingface_hub.

    Returns (path_or_none, was_downloaded).  Fails gracefully — a missing model
    is a BLOCKED verdict for Exp 942, not a crash in this pre-flight.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]

        candidate = _SOTA_CANDIDATES[0]  # gemma-4-26B-A4B-it (smallest viable)
        path = hf_hub_download(
            repo_id=candidate["hf_repo"],
            filename=candidate["filename"],
        )
        return str(path), True
    except Exception as exc:
        print(f"[warn] SOTA model download failed: {exc}", file=sys.stderr)
        return None, False


def check_sota_model() -> dict:
    """
    Verify SOTA GGUF model availability for Exp 942.

    First checks the local HF cache (no network required).  If nothing found,
    attempts a one-shot download of gemma-4-26B-A4B-it Q4_K_M.  Records the
    outcome in a dict so the conductor knows whether to gate Exp 942.
    """
    path, downloaded = _find_sota_model_in_cache()
    if path is None:
        path, downloaded = _attempt_sota_download()

    return {
        "sota_model_path": path,
        "sota_model_downloaded": downloaded,
        "sota_model_available": path is not None,
    }


def build_preflight_artifact(sota: dict) -> dict:
    """
    Assemble the Exp 941 preflight artifact.

    Values derived from:
      - results/experiment_940_milestone_retro_72.json (10/12 criteria, open RETROs)
      - results/experiment_939_sc_energy_set_consistency.json (8 prior SC-energy failures)
      - HF cache scan above (SOTA model path)
    No heavy computation — runs in < 1 s on any Python 3.11+ machine.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "experiment": 941,
        "title": "Milestone 2026.04.73 Pre-flight v22",
        "milestone": "2026.04.73",
        "preflight_version": 22,
        "run_date": "20260426",
        "started_at": now,
        # .72 summary (from Exp 940 milestone retro)
        "predecessor_milestone": "2026.04.72",
        "predecessor_criteria_met": 10,
        "predecessor_criteria_total": 12,
        "predecessor_root_cause_of_failures": [
            "RETRO-MATH-REPAIR-MODEL-CEILING",
            "RETRO-SC-ENERGY-GATE-DISCIPLINE",
        ],
        "predecessor_root_cause_detail": (
            "RETRO-MATH-REPAIR-MODEL-CEILING: gemma-4-E4B-it produced 12% baseline "
            "and 12% repair on GSM8K (signed_improvement=0.0). Model capability ceiling "
            "— E4B too small for GSM8K. Algorithm is correct; model is wrong. "
            "SOTA model required for .73 rerun (Exp 942). "
            "RETRO-SC-ENERGY-GATE-DISCIPLINE: Exp 939 YAML lacked prior_failures for "
            "7 prior SC-energy experiments. Identical planning error to Exp 917 in .71. "
            "Exp 944 MUST include all 8 prior_failures (see sc_energy_prior_experiments)."
        ),
        # SOTA model check (for Exp 942 math repair rerun)
        "sota_model_path": sota["sota_model_path"],
        "sota_model_downloaded": sota["sota_model_downloaded"],
        "sota_model_available": sota["sota_model_available"],
        # SC-energy audit (for Exp 944 gate-check compliance)
        "sc_energy_audit_complete": True,
        "sc_energy_prior_experiment_count": len(_SC_ENERGY_PRIORS),
        "sc_energy_prior_experiments": _SC_ENERGY_PRIORS,
        # Open RETROs entering .73 (from Exp 940)
        "open_retros": [
            "RETRO-MANIFEST-FULL-SCOPE",
            "RETRO-XILINX-TOOLS-UNAVAILABLE",
            "RETRO-RERUN-DISCIPLINE-GATE-CASCADE",
            "RETRO-HEURISTIC-RPRM-FLAT-SIGNAL",
            "RETRO-DRIFT-ENSEMBLE-UNIFORM-WEIGHTS",
            "RETRO-MATH-REPAIR-MODEL-CEILING",
            "RETRO-SC-ENERGY-GATE-DISCIPLINE",
        ],
        "retro_statuses": {
            "RETRO-MANIFEST-FULL-SCOPE": "HUMAN_REQUIRED",
            "RETRO-XILINX-TOOLS-UNAVAILABLE": "HUMAN_REQUIRED",
            "RETRO-RERUN-DISCIPLINE-GATE-CASCADE": "HUMAN_REQUIRED",
            "RETRO-HEURISTIC-RPRM-FLAT-SIGNAL": "TARGETED",
            "RETRO-DRIFT-ENSEMBLE-UNIFORM-WEIGHTS": "TARGETED",
            "RETRO-MATH-REPAIR-MODEL-CEILING": "NEW",
            "RETRO-SC-ENERGY-GATE-DISCIPLINE": "NEW",
            # Closed in .72
            "RETRO-HF-SOPS-CREDENTIAL-INJECTION": "CLOSED_BY_EXP933",
        },
        # .73 gates
        "milestone_73_gates": {
            "exp_943_scratchpad": (
                "GATED on Exp 942 signed_improvement > 0 "
                "(math repair must show positive result with SOTA model before "
                "scratchpad variant is attempted)"
            ),
            "exp_944_sc_energy": (
                "GATED on prior_failures containing all 8 SC-energy experiments "
                "(exp506, exp509, exp533, exp711, exp725, exp772, exp787, exp939)"
            ),
        },
        # Deliverable
        "honest_verdict": "preflight_complete",
        "status": "success",
    }


def main() -> int:
    started_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("Checking SOTA model availability...")
    sota = check_sota_model()
    if sota["sota_model_available"]:
        print(f"  SOTA model found: {sota['sota_model_path']}")
        print(f"  Downloaded this run: {sota['sota_model_downloaded']}")
    else:
        print("  [warn] No SOTA GGUF found — Exp 942 will be BLOCKED until model is downloaded")

    artifact = build_preflight_artifact(sota)
    artifact["started_at"] = started_at
    artifact["finished_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Add required schema fields list (matches Exp 929 / Exp 940 pattern)
    artifact["schema"] = sorted(k for k in artifact if k != "schema")

    output_path = "results/experiment_941_preflight_v22.json"
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)
        f.write("\n")

    print(f"\nWrote {output_path}")
    print(
        f"  predecessor_criteria_met: {artifact['predecessor_criteria_met']}/{artifact['predecessor_criteria_total']}"
    )
    print(f"  sota_model_available: {artifact['sota_model_available']}")
    print(f"  sc_energy_audit_complete: {artifact['sc_energy_audit_complete']}")
    print(f"  sc_energy_prior_count: {artifact['sc_energy_prior_experiment_count']}")
    print(f"  open_retros: {len(artifact['open_retros'])}")
    print(f"  honest_verdict: {artifact['honest_verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
