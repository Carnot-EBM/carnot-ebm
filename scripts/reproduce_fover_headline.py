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
import sys
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Published acceptance CI
# ---------------------------------------------------------------------------

CONDITION_A_CI_LOW = 0.9027
CONDITION_A_CI_HIGH = 0.9235
LEARNING_CONTRIB_CI_LOW = 0.0125
LEARNING_CONTRIB_CI_HIGH = 0.0245

RANDOM_SEEDS: tuple[int, ...] = (42, 137, 271, 314, 1729)
N_EXAMPLES = 1000


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
