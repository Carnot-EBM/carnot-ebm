"""Exp 2837 MBPP ensemble-v7b retry artifact.

This module intentionally reuses the Exp 2829 MBPP evaluator contract. The
retry exists to write the operator-requested
``results/experiment_2837_mbpp_ensemble_eval.json`` path while preserving the
same anti-fabrication boundary: live resources are checked before candidate
generation, and missing resources produce ``blocked_<resource>`` with null
metrics rather than plausible-looking benchmark numbers.

Spec: REQ-VERIFY-MBPP-2837, SCENARIO-VERIFY-MBPP-2837.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from carnot.eval import mbpp_ensemble_v7b as base


OUTPUT_FILENAME = "experiment_2837_mbpp_ensemble_eval.json"
REPO_ROOT = base.REPO_ROOT

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "corpus",
    "n_problems",
    "n_seeds",
    "condition_a_production_auroc_mean",
    "condition_a_production_auroc_std",
    "condition_b_architecture_only_auroc_mean",
    "condition_b_architecture_only_auroc_std",
    "learning_contribution",
    "per_verifier_condition_a_auroc",
    "per_verifier_condition_b_auroc",
    "vanilla_qwen36_pass_at_1",
    "random_seeds_used",
    "reproducibility_checksum",
    "model_specs",
    "duration_s",
    "preconditions_checked",
    "fr11_state_files",
    "state_files_restored_sha_match",
    "methodology_note",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix.",
    "corpus": "Identifies corpus.",
    "n_problems": "Sample size and exact MBPP subset size.",
    "n_seeds": "Replication count across adversarial seeds.",
    "condition_a_production_auroc_mean": "Production FR-11 headline AUROC.",
    "condition_a_production_auroc_std": "Production AUROC replication noise.",
    "condition_b_architecture_only_auroc_mean": "Architecture-only baseline AUROC.",
    "condition_b_architecture_only_auroc_std": "Architecture-only replication noise.",
    "learning_contribution": "Condition A minus Condition B memory contribution.",
    "per_verifier_condition_a_auroc": "Per-verifier production attribution.",
    "per_verifier_condition_b_auroc": "Per-verifier architecture-only attribution.",
    "vanilla_qwen36_pass_at_1": "Generator baseline before verifier ranking.",
    "random_seeds_used": "Deterministic replay of the five-seed protocol.",
    "reproducibility_checksum": "Content-addressed guard against drift.",
    "model_specs": "Names the mandated Qwen3.6 GGUF compute target.",
    "duration_s": "Real wall-clock measurement; sleep-padding forbidden.",
    "preconditions_checked": "Records resources checked before inference.",
    "fr11_state_files": "Names the state isolated for Condition B.",
    "state_files_restored_sha_match": "Proves non-destructive state restore.",
    "methodology_note": "Explains how to interpret measured or blocked output.",
}


def _default_value_for_missing_field(field: str) -> object:
    if field in {"per_verifier_condition_a_auroc", "per_verifier_condition_b_auroc"}:
        return {}
    return None


def _apply_exp2837_contract(artifact: dict[str, object]) -> dict[str, object]:
    contracted = dict(artifact)
    contracted["artifact"] = "experiment_2837_mbpp_ensemble_eval"
    contracted["schema"] = "carnot.mbpp_ensemble_v7b.exp2837"
    contracted["field_principles"] = FIELD_PRINCIPLES
    for field in REQUIRED_ARTIFACT_FIELDS:
        contracted.setdefault(field, _default_value_for_missing_field(field))
    return contracted


def write_artifact(results_dir: Path, artifact: dict[str, object]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / OUTPUT_FILENAME).write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_experiment(
    config: base.ExperimentConfig | None = None,
    *,
    precondition_probe: Callable[
        [base.ExperimentConfig, Sequence[dict[str, object]]], list[base.PreconditionCheck]
    ] = base.probe_preconditions,
    measurement_runner: Callable[
        [base.ExperimentConfig, Sequence[dict[str, object]]], Sequence[base.SeedMeasurement]
    ] = base.default_live_measurement_runner,
    write: bool = True,
) -> dict[str, object]:
    """Run the MBPP retry or write an honest blocked Exp 2837 artifact."""

    config = config or base.ExperimentConfig()
    artifact = base.run_experiment(
        config,
        precondition_probe=precondition_probe,
        measurement_runner=measurement_runner,
        write=False,
    )
    artifact = _apply_exp2837_contract(artifact)
    if write:
        write_artifact(config.output_dir(), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--n-problems", type=int, default=base.DEFAULT_N_PROBLEMS)
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root)
    run_experiment(
        base.ExperimentConfig(
            repo_root=repo_root,
            results_dir=Path(args.results_dir) if args.results_dir else repo_root / "results",
            n_problems=args.n_problems,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
