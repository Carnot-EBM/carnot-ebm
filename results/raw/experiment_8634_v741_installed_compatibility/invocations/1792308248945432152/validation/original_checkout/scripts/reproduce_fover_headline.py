"""Self-contained FoVer headline reproducer for the G2 independence gate.

This script recomputes the FoVer dual-condition AUROC headline from the
labeled corpus without reading any existing results/experiment_2837*.json
artifact. A fresh-clone non-operator can run it after ``pip install -e .``
to independently verify the claim.

The key difference from ``run_experiment`` in fover_memory_leakage_v3 is that
this reproducer:

  - Uses in-process scoring (no GGUF model subprocess required — CPU-only).
  - Uses a minimal precondition probe that skips the SOTA-runtime check
    (which gates on a locally-cached 35B GGUF that external reproducers
    usually won't have).
  - Handles the case where FR-11 state files are absent by reporting an
    honest ``blocked_fr11_state_files`` verdict rather than scoring both
    conditions as architecture-only (which would give misleading numbers).

Usage::

    python3 scripts/reproduce_fover_headline.py

Acceptance CI (from the published experiment, seed set [42,137,271,314,1729]):
  - condition A (production) mean AUROC ∈ [0.9027, 0.9235]
  - learning contribution mean ∈ [0.0125, 0.0245]

Running on a machine WITHOUT FR-11 session-memory state files will yield a
blocked verdict.  The operator can share the ``data/fr11_*.jsonl`` corpus and
``results/session_memory_*/`` state to allow an external reproducer to also
measure condition A.  Condition B (architecture-only) does NOT require state
files and should yield ≈ 0.8947 on any machine with the committed corpus.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Published acceptance CI
# ---------------------------------------------------------------------------

CONDITION_A_CI_LOW = 0.9027
CONDITION_A_CI_HIGH = 0.9235
LEARNING_CONTRIB_CI_LOW = 0.0125
LEARNING_CONTRIB_CI_HIGH = 0.0245

RANDOM_SEEDS: tuple[int, ...] = (42, 137, 271, 314, 1729)
N_EXAMPLES = 1000
EXP3680_CANDIDATE_REL_PATH = Path(
    "results/experiment_3680_dependency_aware_dual_condition_integrity.json"
)


def _ensure_python_path(repo_root: Path) -> None:
    """Add the repo's python/ directory to sys.path if not already present."""
    python_dir = str(repo_root / "python")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)


def build_reproducer_config(
    repo_root: Path,
    seeds: Sequence[int] = RANDOM_SEEDS,
    n_examples: int = N_EXAMPLES,
) -> Any:
    """Return an ExperimentConfig suitable for in-process reproduction."""
    from carnot.eval.fover_memory_leakage_v3 import ExperimentConfig  # type: ignore[import]

    return ExperimentConfig(
        repo_root=repo_root,
        random_seeds=tuple(seeds),
        n_examples=n_examples,
        # Point exp2836_path at a non-existent path — the custom precondition
        # probe below never checks for it, so this prevents any accidental read
        # of the operator's exp2836 artifact.
        exp2836_path=repo_root / "results" / "_g2_reproducer_stub_exp2836.json",
    )


def passthrough_precondition_probe(
    config: Any,
    state_files: Sequence[dict[str, object]],
    model_specs: dict[str, object],
) -> list[Any]:
    """Minimal precondition probe: corpus + FR-11 state files only.

    Why: the standard probe_preconditions also checks for a SOTA GGUF model
    cached locally and an exp2836 preflight artifact.  Both of those are
    irrelevant for the CPU-only verifier-scoring path, but they would block
    every external reproducer who hasn't run exp2836 first.  We strip those
    checks out here and keep only the ones that genuinely gate the scoring.
    """
    from carnot.eval.fover_memory_leakage_v3 import PreconditionCheck, _line_count  # type: ignore[import]

    fover_path = config.repo_root / "data" / "fover_corpus.jsonl"
    checks: list[Any] = []
    if fover_path.exists():
        n_rows = _line_count(fover_path)
        checks.append(
            PreconditionCheck(
                "fover_corpus",
                n_rows >= config.n_examples,
                f"line_count={n_rows}; required>={config.n_examples}",
            )
        )
    else:
        checks.append(PreconditionCheck("fover_corpus", False, "missing"))

    # FR-11 state files are required for condition A (production).  Without
    # them condition A == condition B and learning_contribution ≈ 0, which
    # is outside the published CI and would be a misleading "pass".
    checks.append(
        PreconditionCheck(
            "fr11_state_files",
            bool(state_files),
            f"count={len(state_files)}",
        )
    )
    return checks


def in_process_condition_runner(
    config: Any,
    selected_python: str,  # ignored — we run in-process
    seed: int,
    condition: str,
    require_no_state: bool,
) -> Any:
    """Score one FoVer condition in the current Python process.

    Why in-process instead of subprocess: the subprocess path (used by
    run_experiment by default) requires selected_python to be a real
    executable pointing at an installation with the carnot package on sys.path.
    An external reproducer running from a fresh venv may not have that path
    pre-populated.  Running in-process avoids the issue entirely.
    """
    from carnot.eval.fover_memory_leakage_v3 import score_fover_subset  # type: ignore[import]

    return score_fover_subset(
        repo_root=config.repo_root,
        seed=seed,
        n_examples=config.n_examples,
        condition=condition,
        require_no_state=require_no_state,
    )


def check_acceptance_ci(result: dict[str, Any]) -> tuple[bool, bool]:
    """Return (cond_a_in_ci, lc_in_ci) for the given result dict.

    Both must be True for the reproduction to count as a CI pass.
    """
    cond_a = result.get("condition_a_production_auroc_mean")
    lc_mean_raw = result.get("learning_contribution_ci95")
    if isinstance(lc_mean_raw, dict):
        lc = lc_mean_raw.get("mean")
    else:
        lc = result.get("learning_contribution")

    cond_a_in_ci: bool = (
        cond_a is not None
        and CONDITION_A_CI_LOW <= float(cond_a) <= CONDITION_A_CI_HIGH
    )
    lc_in_ci: bool = (
        lc is not None
        and LEARNING_CONTRIB_CI_LOW <= float(lc) <= LEARNING_CONTRIB_CI_HIGH
    )
    return cond_a_in_ci, lc_in_ci


def _round_metric(value: float | int, digits: int = 6) -> float:
    return round(float(value), digits)


def _seed_t_ci95(values: Sequence[float]) -> dict[str, float]:
    """Return the same small-n seed-summary CI shape used by the frozen path."""

    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("at least one seed value is required")
    mean = sum(numeric) / len(numeric)
    if len(numeric) < 2:
        return {
            "mean": _round_metric(mean),
            "low": _round_metric(mean),
            "high": _round_metric(mean),
        }
    t_crit_by_n = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    t_crit = t_crit_by_n.get(len(numeric), 1.96)
    sample_std = math.sqrt(sum((value - mean) ** 2 for value in numeric) / (len(numeric) - 1))
    half_width = t_crit * sample_std / math.sqrt(len(numeric))
    return {
        "mean": _round_metric(mean),
        "low": _round_metric(mean - half_width),
        "high": _round_metric(mean + half_width),
    }


def dependency_aware_candidate_bounds_from_artifact(
    exp3680_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Return Exp 3680-derived assertion bounds for the candidate re-freeze."""

    production_ci = exp3680_artifact.get("production_auroc_ci")
    if not isinstance(production_ci, Mapping):
        raise ValueError("Exp 3680 artifact is missing production_auroc_ci")
    production_bounds = list(production_ci.get("ci95") or [])
    if len(production_bounds) != 2:
        raise ValueError("Exp 3680 production_auroc_ci must contain two bounds")

    per_seed = list(exp3680_artifact.get("per_seed_results") or [])
    learning_values = [
        float(dict(row)["learning_contribution_dependency_aware"])
        for row in per_seed
        if "learning_contribution_dependency_aware" in dict(row)
    ]
    if not learning_values:
        raise ValueError("Exp 3680 per_seed_results lack learning contribution values")
    learning_ci = _seed_t_ci95(learning_values)
    learning_point = exp3680_artifact.get("learning_contribution_dependency_aware")
    if learning_point is None:
        learning_point = learning_ci["mean"]

    return {
        "production_auroc_dependency_aware": {
            "point": _round_metric(float(production_ci.get("point"))),
            "headline_candidate_point": _round_metric(
                float(exp3680_artifact.get("production_auroc_dependency_aware"))
            ),
            "ci95": [_round_metric(production_bounds[0]), _round_metric(production_bounds[1])],
            "source": EXP3680_CANDIDATE_REL_PATH.as_posix(),
        },
        "learning_contribution_dependency_aware": {
            "point": _round_metric(float(learning_point)),
            "ci95": [learning_ci["low"], learning_ci["high"]],
            "ci_source": "derived_from_exp3680_per_seed_results_seed_t_ci95",
            "source": EXP3680_CANDIDATE_REL_PATH.as_posix(),
        },
    }


def load_dependency_aware_candidate_bounds(repo_root: Path) -> dict[str, Any]:
    """Load Exp 3680 and derive candidate assertion bounds from it."""

    path = Path(repo_root) / EXP3680_CANDIDATE_REL_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dependency_aware_candidate_bounds_from_artifact(payload)


def check_dependency_aware_candidate_ci(
    result: Mapping[str, Any],
    bounds: Mapping[str, Any],
) -> tuple[bool, bool]:
    """Return (production_auroc_in_ci, learning_contribution_in_ci)."""

    production = result.get("production_auroc_dependency_aware")
    learning = result.get("learning_contribution_dependency_aware")
    production_ci = dict(bounds.get("production_auroc_dependency_aware") or {}).get("ci95") or []
    learning_ci = dict(bounds.get("learning_contribution_dependency_aware") or {}).get("ci95") or []
    production_in_ci = (
        production is not None
        and len(production_ci) == 2
        and float(production_ci[0]) <= float(production) <= float(production_ci[1])
    )
    learning_in_ci = (
        learning is not None
        and len(learning_ci) == 2
        and float(learning_ci[0]) <= float(learning) <= float(learning_ci[1])
    )
    return bool(production_in_ci), bool(learning_in_ci)


def run_dependency_aware_candidate_reproduction(
    repo_root: Path,
    seeds: Sequence[int] = RANDOM_SEEDS,
    n_examples: int = N_EXAMPLES,
) -> dict[str, Any]:
    """Recompute and assert the Exp 3680 dependency-aware candidate additively.

    The default frozen 0.9131 path remains ``run_reproduction`` and is unchanged.
    This function is an opt-in candidate path for an operator re-freeze package.
    """

    root = Path(repo_root)
    _ensure_python_path(root)
    from carnot.verify import dependency_aware_dual_condition_integrity as exp3680  # type: ignore[import]

    started_s = time.time()
    result = dict(
        exp3680.build_artifact(
            root,
            started_s=started_s,
            n_examples=n_examples,
            random_seeds=tuple(seeds),
            bootstrap_seeds=tuple(seeds),
            adversarial_verify_clean=True,
        )
    )
    bounds = load_dependency_aware_candidate_bounds(root)
    if result.get("per_seed_results"):
        learning_values = [
            float(row["learning_contribution_dependency_aware"])
            for row in result["per_seed_results"]
        ]
        result["learning_contribution_ci95"] = _seed_t_ci95(learning_values)
    production_in_ci, learning_in_ci = check_dependency_aware_candidate_ci(result, bounds)
    result["candidate_exp3680_assertion_bounds"] = bounds
    result["candidate_production_auroc_in_exp3680_ci"] = production_in_ci
    result["candidate_learning_contribution_in_exp3680_ci"] = learning_in_ci
    result["candidate_reproduction_asserts_in_ci"] = bool(production_in_ci and learning_in_ci)
    return result


def run_reproduction(
    repo_root: Path,
    seeds: Sequence[int] = RANDOM_SEEDS,
    n_examples: int = N_EXAMPLES,
) -> dict[str, Any]:
    """Run the FoVer dual-condition recompute and return the result dict.

    Does NOT read results/experiment_2837*.json.  Does NOT write any artifact.
    Raises ImportError if the carnot package cannot be imported.
    """
    _ensure_python_path(repo_root)
    from carnot.eval.fover_memory_leakage_v3 import run_experiment  # type: ignore[import]

    cfg = build_reproducer_config(repo_root, seeds, n_examples)
    return run_experiment(
        cfg,
        precondition_probe=passthrough_precondition_probe,
        condition_runner=in_process_condition_runner,
        write=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    run_candidate = "--dependency-aware-candidate" in args
    repo_root = Path(__file__).resolve().parent.parent
    _ensure_python_path(repo_root)

    # -- precondition: corpus
    corpus = repo_root / "data" / "fover_corpus.jsonl"
    if not corpus.exists():
        print("BLOCKED: data/fover_corpus.jsonl not found", file=sys.stderr)
        return 1

    # -- precondition: eval module importable
    try:
        from carnot.eval import fover_memory_leakage_v3 as _m  # noqa: F401
    except ImportError as exc:
        print(f"BLOCKED: cannot import carnot.eval.fover_memory_leakage_v3: {exc}", file=sys.stderr)
        return 1

    if run_candidate:
        result = run_dependency_aware_candidate_reproduction(repo_root)
        if str(result.get("honest_verdict", "")).startswith("complete: blocked"):
            print(f"BLOCKED: {result.get('honest_verdict')}", file=sys.stderr)
            return 1
        bounds = result.get("candidate_exp3680_assertion_bounds", {})
        production = result.get("production_auroc_dependency_aware")
        learning = result.get("learning_contribution_dependency_aware")
        checksum = result.get("reproducibility_checksum")
        production_in_ci, learning_in_ci = check_dependency_aware_candidate_ci(result, bounds)
        production_ci = dict(bounds.get("production_auroc_dependency_aware") or {}).get("ci95")
        learning_ci = dict(bounds.get("learning_contribution_dependency_aware") or {}).get("ci95")

        print(f"dependency-aware production mean AUROC: {production:.6f}")
        print(f"dependency-aware learning contribution: {learning:.6f}")
        print(f"reproducibility_checksum:              {checksum}")
        print()
        print(f"production AUROC in exp3680 CI {production_ci}: {production_in_ci}")
        print(f"learning contribution in exp3680 CI {learning_ci}: {learning_in_ci}")
        if production_in_ci and learning_in_ci:
            print("\nRESULT: PASS — dependency-aware candidate reproduces within exp3680 CI")
            return 0
        print("\nRESULT: FAIL — dependency-aware candidate outside exp3680 CI")
        return 1

    result = run_reproduction(repo_root)

    verdict = result.get("honest_verdict", "")
    if str(verdict).startswith("blocked"):
        blocked = result.get("blocked_resources", [])
        print(f"BLOCKED: {verdict} — missing: {blocked}", file=sys.stderr)
        return 1

    cond_a = result.get("condition_a_production_auroc_mean")
    cond_b = result.get("condition_b_architecture_only_auroc_mean")
    lc_ci = result.get("learning_contribution_ci95", {})
    lc = lc_ci.get("mean") if isinstance(lc_ci, dict) else result.get("learning_contribution")
    checksum = result.get("reproducibility_checksum")

    print(f"condition A (production)        mean AUROC: {cond_a:.4f}")
    print(f"condition B (architecture-only) mean AUROC: {cond_b:.4f}")
    print(f"learning contribution:                      {lc:.4f}")
    print(f"reproducibility_checksum:                   {checksum}")

    cond_a_in_ci, lc_in_ci = check_acceptance_ci(result)
    print()
    print(f"condition A in CI [{CONDITION_A_CI_LOW}, {CONDITION_A_CI_HIGH}]: {cond_a_in_ci}")
    print(f"learning_contribution in CI [{LEARNING_CONTRIB_CI_LOW}, {LEARNING_CONTRIB_CI_HIGH}]: {lc_in_ci}")

    if cond_a_in_ci and lc_in_ci:
        print("\nRESULT: PASS — FoVer headline reproduces within published CI")
        return 0

    print("\nRESULT: FAIL — one or more numbers outside published CI")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
