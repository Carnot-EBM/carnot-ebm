"""Exp 2838 HumanEval-full ensemble-v7b retry artifact.

This wrapper reuses the existing HumanEval dual-condition evaluator while
targeting the operator-requested
``results/experiment_2838_humaneval_full_ensemble_eval.json`` path. It adds the
explicit ``.venv/bin/python3`` CUDA precondition from the task prompt and keeps
the anti-fabrication boundary intact: missing live resources produce a
``blocked_<resource>`` artifact with null metrics, not inferred results.

Spec: REQ-VERIFY-HUMANEVAL-2838,
      SCENARIO-VERIFY-HUMANEVAL-2838-BLOCKED,
      SCENARIO-VERIFY-HUMANEVAL-2838-LIVE.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from carnot.eval import humaneval_dual_condition_v3 as base


OUTPUT_FILENAME = "experiment_2838_humaneval_full_ensemble_eval.json"
REPO_ROOT = base.REPO_ROOT
ExperimentConfig = base.ExperimentConfig
PreconditionCheck = base.PreconditionCheck
SeedEvaluation = base.SeedEvaluation
REQUIRED_QWEN36_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"

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
    "pass_at_1_vanilla",
    "pass_at_1_after_carnot_correct_production",
    "pass_at_1_after_carnot_correct_architecture_only",
    "vanilla_qwen36_pass_at_1",
    "peer_humaneval_verifier_baselines",
    "peer_baseline_comparison",
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
    "n_problems": "Full HumanEval problem count.",
    "n_seeds": "Replication count across adversarial seeds.",
    "condition_a_production_auroc_mean": "Production FR-11 headline AUROC.",
    "condition_a_production_auroc_std": "Production AUROC replication noise.",
    "condition_b_architecture_only_auroc_mean": "Architecture-only baseline AUROC.",
    "condition_b_architecture_only_auroc_std": "Architecture-only replication noise.",
    "learning_contribution": "Condition A minus Condition B memory contribution.",
    "per_verifier_condition_a_auroc": "Per-verifier production attribution.",
    "per_verifier_condition_b_auroc": "Per-verifier architecture-only attribution.",
    "pass_at_1_vanilla": "Generator baseline before Carnot verifier ranking.",
    "pass_at_1_after_carnot_correct_production": (
        "Pass@1 after production Carnot verifier ranking."
    ),
    "pass_at_1_after_carnot_correct_architecture_only": (
        "Pass@1 after architecture-only verifier ranking."
    ),
    "vanilla_qwen36_pass_at_1": "Exp 2837-compatible alias for vanilla pass@1.",
    "peer_humaneval_verifier_baselines": "External HumanEval verifier comparison inputs.",
    "peer_baseline_comparison": "Comparison against peer HumanEval verifier baselines.",
    "random_seeds_used": "Deterministic replay of the five-seed protocol.",
    "reproducibility_checksum": "Content-addressed guard against drift.",
    "model_specs": "Names the mandated Qwen3.6 GGUF compute target.",
    "duration_s": "Real wall-clock measurement; sleep-padding forbidden.",
    "preconditions_checked": (
        "Records .venv/bin/python3 CUDA, Qwen3.6 cache, dataset, sandbox, and FR-11 checks."
    ),
    "fr11_state_files": "Names the state isolated for Condition B.",
    "state_files_restored_sha_match": "Proves non-destructive state restore.",
    "methodology_note": "Explains how to interpret measured or blocked output.",
}

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _default_value_for_missing_field(field: str) -> object:
    if field in {
        "per_verifier_condition_a_auroc",
        "per_verifier_condition_b_auroc",
        "peer_humaneval_verifier_baselines",
    }:
        return {}
    return None


def _field_satisfied_by(artifact: dict[str, object]) -> str:
    verdict = str(artifact.get("honest_verdict", ""))
    if verdict.startswith(("complete:", "success:")):
        return "measured output"
    return "blocked before measurement"


def _pass_metric(artifact: dict[str, object], key: str) -> object:
    pass_at_1 = artifact.get("pass_at_1")
    if isinstance(pass_at_1, dict):
        return pass_at_1.get(key)
    return None


def _apply_exp2838_contract(artifact: dict[str, object]) -> dict[str, object]:
    contracted = dict(artifact)
    contracted["artifact"] = "experiment_2838_humaneval_full_ensemble_eval"
    contracted["schema"] = "carnot.humaneval_full_ensemble_eval.exp2838"
    contracted["corpus"] = base.CORPUS
    contracted["n_problems"] = int(contracted.get("n_problems") or contracted.get("n_tasks") or 164)
    contracted["pass_at_1_vanilla"] = _pass_metric(contracted, "vanilla_mean")
    contracted["pass_at_1_after_carnot_correct_production"] = _pass_metric(
        contracted, "condition_a_ranked_mean"
    )
    contracted["pass_at_1_after_carnot_correct_architecture_only"] = _pass_metric(
        contracted, "condition_b_ranked_mean"
    )
    contracted["vanilla_qwen36_pass_at_1"] = contracted["pass_at_1_vanilla"]
    contracted.setdefault("peer_humaneval_verifier_baselines", {})
    contracted.setdefault("peer_baseline_comparison", None)
    contracted["field_principles"] = FIELD_PRINCIPLES
    for field in REQUIRED_ARTIFACT_FIELDS:
        contracted.setdefault(field, _default_value_for_missing_field(field))
    satisfied_by = _field_satisfied_by(contracted)
    contracted["field_provenance"] = {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "satisfied_by": satisfied_by,
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    return contracted


def _venv_python3_cuda_check(
    config: base.ExperimentConfig,
    *,
    command_runner: CommandRunner = subprocess.run,
) -> base.PreconditionCheck:
    command_text = '.venv/bin/python3 -c "import torch; assert torch.cuda.is_available()"'
    command = [
        str(config.repo_root / ".venv" / "bin" / "python3"),
        "-c",
        "import torch; assert torch.cuda.is_available()",
    ]
    try:
        proc = command_runner(
            command,
            capture_output=True,
            text=True,
            timeout=config.probe_timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - host dependent
        return base.PreconditionCheck("cuda", False, f"{command_text} failed: {exc}")
    detail = (proc.stdout or proc.stderr or f"returncode={proc.returncode}").strip()
    if proc.returncode == 0:
        return base.PreconditionCheck("cuda", True, f"{command_text} passed; {detail}")
    return base.PreconditionCheck("cuda", False, f"{command_text} failed; {detail}")


def _qwen36_cache_check(model_specs: dict[str, object]) -> base.PreconditionCheck:
    paths: list[str] = []
    selected_model_hf_id = model_specs.get("selected_model_hf_id")
    if selected_model_hf_id is not None and selected_model_hf_id != REQUIRED_QWEN36_HF_ID:
        return base.PreconditionCheck(
            "qwen36_gguf_cache",
            False,
            f"selected_model_hf_id must be {REQUIRED_QWEN36_HF_ID}, got {selected_model_hf_id}",
        )
    selected_model_path = model_specs.get("selected_model_path")
    if selected_model_hf_id == REQUIRED_QWEN36_HF_ID and selected_model_path:
        paths.append(str(selected_model_path))
    for row in model_specs.get("sota_models_cached", []):
        if isinstance(row, dict) and row.get("hf_id") == REQUIRED_QWEN36_HF_ID:
            for key in ("path", "resolved_path", "model_path"):
                if row.get(key):
                    paths.append(str(row[key]))

    cache_root = (
        Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
    )
    paths.extend(str(path) for path in cache_root.glob("snapshots/**/*.gguf"))
    real_paths = sorted({path for path in paths if Path(path).is_file()})
    if real_paths:
        return base.PreconditionCheck(
            "qwen36_gguf_cache",
            True,
            f"{REQUIRED_QWEN36_HF_ID} cached at {real_paths[0]}",
        )
    return base.PreconditionCheck(
        "qwen36_gguf_cache",
        False,
        f"no real {REQUIRED_QWEN36_HF_ID} .gguf found in preflight or HF cache",
    )


def probe_preconditions(
    config: base.ExperimentConfig,
    state_files: Sequence[dict[str, object]],
    model_specs: dict[str, object],
    *,
    command_runner: CommandRunner = subprocess.run,
) -> list[base.PreconditionCheck]:
    """Check the prompt-mandated CUDA command before the base HumanEval gates."""

    return [
        _venv_python3_cuda_check(config, command_runner=command_runner),
        _qwen36_cache_check(model_specs),
        *base.probe_preconditions(config, state_files, model_specs),
    ]


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
        [base.ExperimentConfig, Sequence[dict[str, object]], dict[str, object]],
        list[base.PreconditionCheck],
    ] = probe_preconditions,
    measurement_runner: Callable[
        [base.ExperimentConfig, Sequence[dict[str, object]], dict[str, object]],
        Sequence[base.SeedEvaluation],
    ] = base.default_live_measurement_runner,
    write: bool = True,
) -> dict[str, object]:
    """Run the HumanEval retry or write an honest blocked Exp 2838 artifact."""

    config = config or base.ExperimentConfig()
    artifact = base.run_experiment(
        config,
        precondition_probe=precondition_probe,
        measurement_runner=measurement_runner,
        write=False,
    )
    artifact = _apply_exp2838_contract(artifact)
    if write:
        write_artifact(config.output_dir(), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--n-tasks", type=int, default=base.DEFAULT_N_TASKS)
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root)
    run_experiment(
        base.ExperimentConfig(
            repo_root=repo_root,
            results_dir=Path(args.results_dir) if args.results_dir else repo_root / "results",
            n_tasks=args.n_tasks,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
