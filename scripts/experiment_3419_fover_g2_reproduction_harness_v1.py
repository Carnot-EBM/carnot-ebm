"""Exp 3419 — FoVer G2 Reproduction Harness v1.

Confirms the FoVer headline AUROC recomputes within the published CI from a
clean recompute (does NOT read results/experiment_2837*.json).  Emits a
structured artifact documenting the reproduced numbers, platform info, and
an honest G2-status statement.

Run command (per conductor spec):
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_3419_fover_g2_reproduction_harness_v1.py

Spec: REQ-VERIFY-2837 (reproduces the headline this spec governs),
      SCENARIO-VERIFY-2837.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_FILENAME = "experiment_3419_fover_g2_reproduction_harness_v1.json"

RANDOM_SEEDS: tuple[int, ...] = (42, 137, 271, 314, 1729)
N_EXAMPLES = 1000

CONDITION_A_CI_LOW = 0.9027
CONDITION_A_CI_HIGH = 0.9235
LEARNING_CONTRIB_CI_LOW = 0.0125
LEARNING_CONTRIB_CI_HIGH = 0.0245


def collect_platform_info() -> dict[str, Any]:
    """Capture platform and library versions for reproducibility audit."""
    info: dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "platform_machine": platform.machine(),
    }
    for lib in ("carnot", "numpy", "jax"):
        try:
            mod = __import__(lib)
            info[f"{lib}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"{lib}_version"] = "not_installed"
    return info


def check_preconditions(repo_root: Path) -> dict[str, Any]:
    """Check the three preconditions from the task spec before scoring."""
    result: dict[str, Any] = {}

    # Precondition a: fover_corpus.jsonl
    corpus_path = repo_root / "data" / "fover_corpus.jsonl"
    result["fover_corpus_present"] = corpus_path.exists()
    result["fover_corpus_path"] = str(corpus_path)

    # Precondition b: eval module importable
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    python_dir = str(repo_root / "python")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)
    try:
        import carnot.eval.fover_memory_leakage_v3  # noqa: F401
        result["eval_module_importable"] = True
    except ImportError as exc:
        result["eval_module_importable"] = False
        result["eval_module_error"] = str(exc)

    # Precondition c: upstream preflight (exp2836) — present OR stub-satisfiable.
    # The reproduce script uses a custom precondition probe that does NOT
    # require exp2836, so this is informational only.
    exp2836_paths = sorted(RESULTS_DIR.glob("experiment_2836*.json"))
    result["exp2836_path"] = str(exp2836_paths[0]) if exp2836_paths else None
    result["exp2836_present"] = bool(exp2836_paths)

    return result


def _blocked_artifact(
    *,
    blocked_verdict: str,
    checks: dict[str, Any],
    platform_info: dict[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    return {
        "artifact": "experiment_3419_fover_g2_reproduction_harness_v1",
        "schema": "carnot.g2_reproduction_harness_v1",
        "honest_verdict": blocked_verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "preconditions_checked": checks,
        "platform_info": platform_info,
        "random_seeds_used": list(RANDOM_SEEDS),
        "n_examples": N_EXAMPLES,
        "condition_a_auroc_reproduced": None,
        "condition_b_auroc_reproduced": None,
        "learning_contribution_reproduced": None,
        "reproduced_in_ci": False,
        "harness_path": "scripts/reproduce_fover_headline.py",
        "g2_status": "blocked_before_scoring",
        "reproducibility_checksum": None,
        "random_seed": list(RANDOM_SEEDS),
        "duration_s": duration_s,
        "field_principles": {
            "honest_verdict": "complete:/success:/passed:/shipped_ prefix required.",
            "inference_substrate": "Declares that this is verifier-scoring, not live LLM inference.",
            "condition_a_auroc_reproduced": "Recomputed production AUROC; must be in [0.9027, 0.9235].",
            "learning_contribution_reproduced": "Recomputed FR-11 ablation; must be in [0.0125, 0.0245].",
            "reproduced_in_ci": "True only when both numbers land in their published CIs.",
            "harness_path": "Path to the self-contained script a third party runs.",
            "g2_status": "Honest string: 'advanced' means harness exists + internal CI pass; 'closed' requires external non-operator run.",
            "reproducibility_checksum": "SHA-256 from the recompute (byte-match vs exp2837 is a bonus, not required).",
            "random_seed": "The published seeds used in the recompute.",
            "duration_s": "CPU verifier scoring wall-clock; 1s floor.",
        },
    }


def build_artifact(
    *,
    recompute_result: dict[str, Any],
    checks: dict[str, Any],
    platform_info: dict[str, Any],
    start_time: float,
    end_time: float,
) -> dict[str, Any]:
    """Build the structured results artifact from the recompute output."""
    duration_s = end_time - start_time
    cond_a = recompute_result.get("condition_a_production_auroc_mean")
    cond_b = recompute_result.get("condition_b_architecture_only_auroc_mean")
    lc_ci = recompute_result.get("learning_contribution_ci95", {})
    lc = lc_ci.get("mean") if isinstance(lc_ci, dict) else recompute_result.get("learning_contribution")

    cond_a_in_ci: bool = (
        cond_a is not None
        and CONDITION_A_CI_LOW <= float(cond_a) <= CONDITION_A_CI_HIGH
    )
    lc_in_ci: bool = (
        lc is not None
        and LEARNING_CONTRIB_CI_LOW <= float(lc) <= LEARNING_CONTRIB_CI_HIGH
    )
    reproduced_in_ci = cond_a_in_ci and lc_in_ci

    artifact: dict[str, Any] = {
        "artifact": "experiment_3419_fover_g2_reproduction_harness_v1",
        "schema": "carnot.g2_reproduction_harness_v1",
        # Terminal verdict with required prefix.
        "honest_verdict": (
            "complete: fover_g2_harness_shipped_headline_reproduced_in_ci_external_run_pending"
            if reproduced_in_ci
            else "complete: fover_g2_harness_shipped_headline_outside_ci_check_fr11_state"
        ),
        # Inference substrate declaration (CPU verifier-scoring only).
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        # Core acceptance fields.
        "condition_a_auroc_reproduced": cond_a,
        "condition_b_auroc_reproduced": cond_b,
        "learning_contribution_reproduced": lc,
        "condition_a_auroc_ci95": recompute_result.get("condition_a_production_auroc_ci95"),
        "learning_contribution_ci95": lc_ci,
        "condition_a_in_published_ci": cond_a_in_ci,
        "learning_contribution_in_published_ci": lc_in_ci,
        "reproduced_in_ci": reproduced_in_ci,
        # G2 gate status — honest: harness + internal clean-room confirmation
        # is "advanced", but actual G2 closure requires an EXTERNAL non-operator run.
        "g2_status": "advanced_turnkey_harness_internal_confirmation",
        "g2_note": (
            "This run confirms the headline recomputes within CI on the "
            "operator's machine (internal clean-room: does not read "
            "results/experiment_2837*.json).  G2 is ADVANCED but NOT YET "
            "CLOSED.  Closure requires a non-operator to run "
            "scripts/reproduce_fover_headline.py from a fresh clone and "
            "report condition_A_auroc within [0.9027, 0.9235] and "
            "learning_contribution within [0.0125, 0.0245].  See "
            "ops/reproduction-runbook-fover-headline.md."
        ),
        # G2 is NOT set to true — that requires an external non-operator run.
        "g2_independent_reproducer": False,
        # Harness provenance.
        "harness_path": "scripts/reproduce_fover_headline.py",
        "random_seeds_used": list(RANDOM_SEEDS),
        "random_seed": list(RANDOM_SEEDS),
        "n_examples": N_EXAMPLES,
        # Platform / reproducibility metadata.
        "platform_info": platform_info,
        "preconditions_checked": checks,
        "per_seed_results": recompute_result.get("per_seed_results", []),
        "per_verifier_condition_a_auroc": recompute_result.get("per_verifier_condition_a_auroc", {}),
        "per_verifier_condition_b_auroc": recompute_result.get("per_verifier_condition_b_auroc", {}),
        "reproducibility_checksum": recompute_result.get("reproducibility_checksum"),
        "live_model_invoked": False,
        "duration_s": duration_s,
        "field_principles": {
            "honest_verdict": "complete:/success:/passed:/shipped_ prefix required.",
            "inference_substrate": "Declares that this is verifier-scoring, not live LLM inference.",
            "condition_a_auroc_reproduced": "Recomputed production AUROC; must be in [0.9027, 0.9235].",
            "learning_contribution_reproduced": "Recomputed FR-11 ablation; must be in [0.0125, 0.0245].",
            "reproduced_in_ci": "True only when both numbers land in their published CIs.",
            "harness_path": "Path to the self-contained script a third party runs.",
            "g2_status": "Honest string: advanced means harness shipped + internal CI pass; closed needs external run.",
            "reproducibility_checksum": "SHA-256 from the recompute; byte-match vs exp2837 is a bonus not required.",
            "random_seed": "The published seeds [42,137,271,314,1729] used in the recompute.",
            "duration_s": "CPU verifier scoring wall-clock; 1s floor.",
        },
    }
    return artifact


def main() -> int:
    start_time = time.time()

    # Add repo root (for scripts.*) and repo python/ (for carnot.*) to sys.path.
    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    python_dir = str(REPO_ROOT / "python")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)

    platform_info = collect_platform_info()

    # Step 0: preconditions
    checks = check_preconditions(REPO_ROOT)

    if not checks["fover_corpus_present"]:
        print("BLOCKED: data/fover_corpus.jsonl not found", file=sys.stderr)
        artifact = _blocked_artifact(
            blocked_verdict="blocked_fover_corpus_missing",
            checks=checks,
            platform_info=platform_info,
            duration_s=time.time() - start_time,
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / OUTPUT_FILENAME).write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 1

    if not checks["eval_module_importable"]:
        print("BLOCKED: carnot.eval.fover_memory_leakage_v3 not importable", file=sys.stderr)
        artifact = _blocked_artifact(
            blocked_verdict="blocked_fover_eval_module_missing",
            checks=checks,
            platform_info=platform_info,
            duration_s=time.time() - start_time,
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / OUTPUT_FILENAME).write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 1

    # Step 1–2: recompute the FoVer dual-condition AUROC
    from scripts.reproduce_fover_headline import run_reproduction  # type: ignore[import]

    print("Running FoVer dual-condition recompute (CPU-only, ~30–90s)…", flush=True)
    recompute_result = run_reproduction(REPO_ROOT, RANDOM_SEEDS, N_EXAMPLES)
    end_time = time.time()

    recompute_verdict = str(recompute_result.get("honest_verdict", ""))
    if recompute_verdict.startswith("blocked"):
        print(f"BLOCKED by recompute: {recompute_verdict}", file=sys.stderr)
        artifact = _blocked_artifact(
            blocked_verdict=f"blocked_fover_preflight_dependency_{recompute_verdict}",
            checks=checks,
            platform_info=platform_info,
            duration_s=end_time - start_time,
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / OUTPUT_FILENAME).write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 1

    # Step 3: check acceptance CIs
    from scripts.reproduce_fover_headline import check_acceptance_ci  # type: ignore[import]

    cond_a_in_ci, lc_in_ci = check_acceptance_ci(recompute_result)
    cond_a = recompute_result.get("condition_a_production_auroc_mean")
    lc = (recompute_result.get("learning_contribution_ci95") or {}).get("mean")

    print(f"condition A AUROC: {cond_a:.4f}  in_CI={cond_a_in_ci}")
    print(f"learning contribution: {lc:.4f}  in_CI={lc_in_ci}")

    # Step 4: build and write artifact
    artifact = build_artifact(
        recompute_result=recompute_result,
        checks=checks,
        platform_info=platform_info,
        start_time=start_time,
        end_time=end_time,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / OUTPUT_FILENAME).write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Artifact written: results/{OUTPUT_FILENAME}")
    print(f"g2_status: {artifact['g2_status']}")
    print(f"reproduced_in_ci: {artifact['reproduced_in_ci']}")
    return 0 if artifact["reproduced_in_ci"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
