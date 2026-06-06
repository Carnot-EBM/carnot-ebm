#!/usr/bin/env python3
"""Exp 3893: independent replication of the Exp 3882 EBT FUNDAMENTAL result.

Spec refs: REQ-EBT-3893, SCENARIO-EBT-3893-REPLICATION-GATE,
SCENARIO-EBT-3893-SCHEMA, SCENARIO-EBT-3893-REUSE.

This script is intentionally a thin aggregation wrapper around Exp 3882. It
keeps the measurement path fixed and only changes the seed cohort and artifact
schema needed to decide whether the `.359` negative can be banked.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 3893
SCHEMA = "carnot.experiment_3893_ebt_fundamental_replication.v1"
OUTPUT_REL_PATH = Path("results/experiment_3893_ebt_fundamental_replication.json")
EXP3882_REL_PATH = Path("scripts/experiments/experiment_3882_thesis_a_partb_killgate.py")

DEFAULT_SEEDS = (4, 5)
DEFAULT_DIM = 768
DEFAULT_LAYERS = 4
DEFAULT_HEADS = 12
DEFAULT_BLOCK_SIZE = 48
DEFAULT_DIGITS = 3
DEFAULT_TRAIN_STEPS = 12_000
DEFAULT_N_EVAL = 100
DEFAULT_BEAM = 8
DEFAULT_TOPK = 12
DEFAULT_DEVICE = "cuda:0"

BARE_REQUIRED_FIELDS = (
    "replication_outcome",
    "ebt_beam_accuracy_mean",
    "ebt_argmin_accuracy_mean",
    "ar_best_accuracy_mean",
    "n_valid_seeds",
    "matched_flops_ratio",
    "seeds_used",
    "n_heldout",
    "retrained",
    "preconditions_checked",
    "model_specs",
    "random_seed",
    "random_seeds_used",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
)
REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "honest_verdict",
    *BARE_REQUIRED_FIELDS,
    "all_configured_seeds",
    "valid_seeds",
    "invalid_seed_reasons",
    "seed_evaluations",
    "field_principles",
)

FIELD_PRINCIPLES = {
    "replication_outcome": (
        "REPLICATED | REFUTED | INCONCLUSIVE - does the .359 FUNDAMENTAL hold at fresh seeds; "
        "the bank-or-not signal."
    ),
    "ebt_beam_accuracy_mean": "Mean global-beam accuracy across valid seeds - ~0 confirms FUNDAMENTAL.",
    "ebt_argmin_accuracy_mean": "Mean greedy-argmin accuracy across valid seeds - ~0 confirms FUNDAMENTAL.",
    "ar_best_accuracy_mean": (
        "Positive control mean - must be in [0.4,0.95]; AR collapse => INCONCLUSIVE seed."
    ),
    "n_valid_seeds": "Seeds whose AR positive control passed; the replication is meaningful only with >=2.",
    "matched_flops_ratio": "EBT/AR inference-FLOP ratio - a win only counts at equal compute.",
    "seeds_used": "Fresh random seeds actually evaluated.",
    "n_heldout": "Held-out examples evaluated per seed.",
    "retrained": "True only when no evaluated fresh seed reused a checkpoint.",
    "preconditions_checked": "CUDA, scaled-harness import, and Exp 3882 reuse evidence.",
    "model_specs": "Scaled model and decode configuration.",
    "random_seed": "First configured fresh seed.",
    "random_seeds_used": "All evaluated random seeds.",
    "reproducibility_checksum": "SHA256 checksum over configuration and results.",
    "duration_s": "Real wall-clock duration in seconds.",
    "inference_substrate": "Execution substrate for the artifact.",
}


@dataclass(frozen=True)
class PreconditionReport:
    """Pre-launch checks that gate live model work."""

    cuda: bool
    cuda_device_count: int
    scaled_harness_import: bool
    exp3882_pipeline_import: bool
    exp3882_reusable_functions: bool
    python_executable: str = field(default_factory=lambda: sys.executable)
    cuda_devices: list[str] = field(default_factory=list)
    scaled_harness_error: str | None = None
    exp3882_error: str | None = None


@dataclass(frozen=True)
class ReplicationGate:
    """Threshold outcome for the Exp 3893 replication gate."""

    replication_outcome: str
    honest_verdict: str


def load_exp3882_pipeline() -> Any:  # pragma: no cover - live Exp 3882 import path.
    """Load the Exp 3882 measurement module without importing a package path."""

    script_path = REPO_ROOT / EXP3882_REL_PATH
    spec = importlib.util.spec_from_file_location("exp3882_for_exp3893", script_path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib defensive guard.
        raise ImportError(f"cannot load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = (
        "_load_or_train_models",
        "evaluate_seed",
        "ebt_beam_generate",
        "matched_ar_best_of_n",
        "load_scaled_modules",
        "check_preconditions",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:  # pragma: no cover - would mean Exp 3882 changed incompatibly.
        raise AttributeError(f"Exp 3882 missing reusable functions: {', '.join(missing)}")
    return module


def fresh_checkpoint_path(*, seed: int, results_dir: Path) -> Path:
    """Return an Exp 3893 checkpoint filename that does not already exist."""

    base = results_dir / f"experiment_3893_ebt_fundamental_replication_seed{seed}.pt"
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = results_dir / f"experiment_3893_ebt_fundamental_replication_seed{seed}_rerun{index}.pt"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate fresh checkpoint path for seed {seed}")  # pragma: no cover


@contextlib.contextmanager
def fresh_checkpoint_policy(pipeline: Any, *, results_dir: Path) -> Iterator[None]:
    """Patch Exp 3882 checkpoint helpers so fresh seeds cannot reuse checkpoints."""

    original_checkpoint_path = pipeline._checkpoint_path
    original_scaled_artifact_checkpoint = pipeline._scaled_artifact_checkpoint

    def checkpoint_path(seed: int) -> Path:
        return fresh_checkpoint_path(seed=seed, results_dir=results_dir)

    def no_scaled_checkpoint(seed: int) -> None:
        return None

    pipeline._checkpoint_path = checkpoint_path
    pipeline._scaled_artifact_checkpoint = no_scaled_checkpoint
    try:
        yield
    finally:
        pipeline._checkpoint_path = original_checkpoint_path
        pipeline._scaled_artifact_checkpoint = original_scaled_artifact_checkpoint


def _stable_checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _as_plain_dict(item: Any) -> dict[str, Any]:
    if is_dataclass(item):
        return asdict(item)
    if isinstance(item, dict):  # pragma: no cover - convenience for future callers.
        return dict(item)
    return dict(vars(item))  # pragma: no cover - convenience for future callers.


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(float(sum(values) / len(values)), 6)


def _is_valid_seed(evaluation: Any) -> bool:
    return (
        0.4 <= float(evaluation.ar_best_accuracy) <= 0.95
        and evaluation.ebt_argmin_accuracy is not None
        and evaluation.ebt_beam_accuracy is not None
    )


def _invalid_seed_reasons(evaluations: Sequence[Any]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for evaluation in evaluations:
        ar_best = float(evaluation.ar_best_accuracy)
        if not (0.4 <= ar_best <= 0.95):
            reasons[str(evaluation.seed)] = f"ar_positive_control_failed_{ar_best:.3f}"
        elif evaluation.ebt_argmin_accuracy is None or evaluation.ebt_beam_accuracy is None:
            reasons[str(evaluation.seed)] = "ebt_metrics_missing"
    return reasons


def adjudicate_replication(
    *,
    ar_best_accuracy_mean: float | None,
    ebt_argmin_accuracy_mean: float | None,
    ebt_beam_accuracy_mean: float | None,
    n_valid_seeds: int,
) -> ReplicationGate:
    """Apply SCENARIO-EBT-3893-REPLICATION-GATE thresholds."""

    if n_valid_seeds < 2:
        return ReplicationGate(
            replication_outcome="INCONCLUSIVE",
            honest_verdict=f"complete: ebt_fundamental_INCONCLUSIVE_only{n_valid_seeds}_valid_seeds_ar_control_collapsed",
        )
    if ar_best_accuracy_mean is None or ebt_argmin_accuracy_mean is None or ebt_beam_accuracy_mean is None:
        return ReplicationGate(
            replication_outcome="INCONCLUSIVE",
            honest_verdict=f"complete: ebt_fundamental_INCONCLUSIVE_only{n_valid_seeds}_valid_seeds_metrics_missing",
        )
    if (
        ebt_beam_accuracy_mean < 0.2 * ar_best_accuracy_mean
        and ebt_argmin_accuracy_mean < 0.2 * ar_best_accuracy_mean
    ):
        return ReplicationGate(
            replication_outcome="REPLICATED",
            honest_verdict=(
                f"complete: ebt_fundamental_REPLICATED_beam{ebt_beam_accuracy_mean:.3f}_"
                f"argmin{ebt_argmin_accuracy_mean:.3f}_vs_ar{ar_best_accuracy_mean:.3f}_"
                f"nseeds{n_valid_seeds}_energy_as_generator_banked_negative"
            ),
        )
    if ebt_beam_accuracy_mean >= 0.5 * ar_best_accuracy_mean:
        return ReplicationGate(
            replication_outcome="REFUTED",
            honest_verdict=(
                f"complete: ebt_fundamental_REFUTED_beam{ebt_beam_accuracy_mean:.3f}_"
                f"recovers_vs_ar{ar_best_accuracy_mean:.3f}_359_verdict_was_artifact"
            ),
        )
    return ReplicationGate(
        replication_outcome="INCONCLUSIVE",
        honest_verdict=(
            f"complete: ebt_fundamental_INCONCLUSIVE_beam{ebt_beam_accuracy_mean:.3f}_"
            f"argmin{ebt_argmin_accuracy_mean:.3f}_vs_ar{ar_best_accuracy_mean:.3f}_"
            f"thresholds_not_decisive_nseeds{n_valid_seeds}"
        ),
    )


def build_artifact(
    *,
    seed_evaluations: Sequence[Any],
    all_configured_seeds: Sequence[int],
    preconditions: PreconditionReport,
    model_specs: dict[str, Any],
    started_s: float,
    finished_s: float,
    inference_substrate: str,
) -> dict[str, Any]:
    """Build the terminal Exp 3893 replication artifact with bare fields."""

    evaluations = list(seed_evaluations)
    valid = [item for item in evaluations if _is_valid_seed(item)]
    valid_seeds = [int(item.seed) for item in valid]
    seeds_used = [int(item.seed) for item in evaluations]
    ar_mean = _mean([float(item.ar_best_accuracy) for item in valid])
    argmin_mean = _mean([float(item.ebt_argmin_accuracy) for item in valid])
    beam_mean = _mean([float(item.ebt_beam_accuracy) for item in valid])
    ratio_mean = _mean([float(item.matched_flops_ratio) for item in valid if item.matched_flops_ratio is not None])
    gate = adjudicate_replication(
        ar_best_accuracy_mean=ar_mean,
        ebt_argmin_accuracy_mean=argmin_mean,
        ebt_beam_accuracy_mean=beam_mean,
        n_valid_seeds=len(valid),
    )
    invalid_reasons = _invalid_seed_reasons(evaluations)
    n_heldout = int(evaluations[0].n_heldout) if evaluations else 0
    retrained = bool(evaluations) and all(not bool(item.checkpoint_reused) for item in evaluations)
    seed_dicts = [_as_plain_dict(item) for item in evaluations]
    random_seed = int(all_configured_seeds[0]) if all_configured_seeds else None
    checksum_payload = {
        "experiment": EXPERIMENT_ID,
        "configured_seeds": list(all_configured_seeds),
        "seed_evaluations": seed_dicts,
        "valid_seeds": valid_seeds,
        "invalid_seed_reasons": invalid_reasons,
        "model_specs": model_specs,
        "preconditions": asdict(preconditions),
        "gate": asdict(gate),
    }
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "honest_verdict": gate.honest_verdict,
        "replication_outcome": gate.replication_outcome,
        "ebt_beam_accuracy_mean": beam_mean,
        "ebt_argmin_accuracy_mean": argmin_mean,
        "ar_best_accuracy_mean": ar_mean,
        "n_valid_seeds": len(valid),
        "matched_flops_ratio": ratio_mean,
        "seeds_used": seeds_used,
        "n_heldout": n_heldout,
        "retrained": retrained,
        "preconditions_checked": asdict(preconditions),
        "model_specs": dict(model_specs, valid_seeds=valid_seeds),
        "random_seed": random_seed,
        "random_seeds_used": seeds_used,
        "reproducibility_checksum": _stable_checksum(checksum_payload),
        "duration_s": round(float(finished_s - started_s), 2),
        "inference_substrate": inference_substrate,
        "all_configured_seeds": list(all_configured_seeds),
        "valid_seeds": valid_seeds,
        "invalid_seed_reasons": invalid_reasons,
        "seed_evaluations": seed_dicts,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    return artifact


def build_blocked_artifact(
    *,
    honest_verdict: str,
    preconditions: PreconditionReport,
    model_specs: dict[str, Any],
    started_s: float,
    finished_s: float,
) -> dict[str, Any]:
    """Build a terminal blocked artifact with the same bare-field schema."""

    checksum_payload = {
        "experiment": EXPERIMENT_ID,
        "honest_verdict": honest_verdict,
        "preconditions": asdict(preconditions),
        "model_specs": model_specs,
    }
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "honest_verdict": honest_verdict,
        "replication_outcome": "INCONCLUSIVE",
        "ebt_beam_accuracy_mean": None,
        "ebt_argmin_accuracy_mean": None,
        "ar_best_accuracy_mean": None,
        "n_valid_seeds": 0,
        "matched_flops_ratio": None,
        "seeds_used": [],
        "n_heldout": 0,
        "retrained": False,
        "preconditions_checked": asdict(preconditions),
        "model_specs": dict(model_specs),
        "random_seed": None,
        "random_seeds_used": [],
        "reproducibility_checksum": _stable_checksum(checksum_payload),
        "duration_s": round(float(finished_s - started_s), 2),
        "inference_substrate": "blocked_precondition",
        "all_configured_seeds": [],
        "valid_seeds": [],
        "invalid_seed_reasons": {},
        "seed_evaluations": [],
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def validate_artifact(artifact: dict[str, Any]) -> list[str]:
    """Return schema errors that make the Exp 3893 artifact non-terminal."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_FIELDS) - set(artifact))
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    verdict = str(artifact.get("honest_verdict", ""))
    if not (verdict.startswith("complete:") or verdict.startswith("blocked_")):
        errors.append("honest_verdict must start with complete: or blocked_")
    for field_name in BARE_REQUIRED_FIELDS:
        value = artifact.get(field_name)
        if isinstance(value, dict) and {"value", "principle"} <= set(value):
            errors.append(f"{field_name} must be a bare value, not a value/principle wrapper")

    if artifact.get("replication_outcome") not in {"REPLICATED", "REFUTED", "INCONCLUSIVE"}:
        errors.append("replication_outcome must be REPLICATED, REFUTED, or INCONCLUSIVE")
    for metric in (
        "ebt_beam_accuracy_mean",
        "ebt_argmin_accuracy_mean",
        "ar_best_accuracy_mean",
        "matched_flops_ratio",
    ):
        value = artifact.get(metric)
        if value is not None and not isinstance(value, (int, float)):
            errors.append(f"{metric} must be numeric or null")
    if not isinstance(artifact.get("n_valid_seeds"), int) or isinstance(artifact.get("n_valid_seeds"), bool):
        errors.append("n_valid_seeds must be an integer")
    if not isinstance(artifact.get("seeds_used"), list):
        errors.append("seeds_used must be a list")
    if not isinstance(artifact.get("random_seeds_used"), list):
        errors.append("random_seeds_used must be a list")
    if not isinstance(artifact.get("n_heldout"), int) or isinstance(artifact.get("n_heldout"), bool):
        errors.append("n_heldout must be an integer")
    if not isinstance(artifact.get("retrained"), bool):
        errors.append("retrained must be a bare bool")
    if not isinstance(artifact.get("preconditions_checked"), dict):
        errors.append("preconditions_checked must be an object")
    if not isinstance(artifact.get("model_specs"), dict):
        errors.append("model_specs must be an object")
    random_seed = artifact.get("random_seed")
    if random_seed is not None and (not isinstance(random_seed, int) or isinstance(random_seed, bool)):
        errors.append("random_seed must be an integer or null")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        errors.append("reproducibility_checksum must be a sha256 hex string")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s must be non-negative")
    if not isinstance(artifact.get("inference_substrate"), str):
        errors.append("inference_substrate must be a string")
    return errors


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Write sorted JSON so reruns are diffable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_preconditions() -> PreconditionReport:  # pragma: no cover - live preflight.
    """Check CUDA, scaled harness import, and Exp 3882 reuse preconditions."""

    import torch

    cuda = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    device_names = [torch.cuda.get_device_name(i) for i in range(device_count)] if cuda else []
    try:
        pipeline = load_exp3882_pipeline()
        exp3882_import = True
        exp3882_error = None
        reusable = all(
            hasattr(pipeline, name)
            for name in (
                "_load_or_train_models",
                "evaluate_seed",
                "ebt_beam_generate",
                "matched_ar_best_of_n",
            )
        )
        scaled_report = pipeline.check_preconditions()
        scaled_import = bool(scaled_report.scaled_harness_import)
        scaled_error = scaled_report.scaled_harness_error
    except Exception as exc:
        exp3882_import = False
        exp3882_error = repr(exc)
        reusable = False
        scaled_import = False
        scaled_error = repr(exc)
    return PreconditionReport(
        cuda=cuda,
        cuda_device_count=device_count,
        scaled_harness_import=scaled_import,
        exp3882_pipeline_import=exp3882_import,
        exp3882_reusable_functions=reusable,
        cuda_devices=device_names,
        scaled_harness_error=scaled_error,
        exp3882_error=exp3882_error,
    )


def evaluate_replication_seed(  # pragma: no cover - live GPU path.
    *,
    pipeline: Any,
    pb: Any,
    sc: Any,
    seed: int,
    device: Any,
    args: argparse.Namespace,
) -> Any:
    """Run one fresh seed through the Exp 3882 train/eval pipeline."""

    ebt, ar, eval_items, checkpoint, reused, diverged = pipeline._load_or_train_models(
        pb=pb,
        sc=sc,
        seed=seed,
        device=device,
        digits=args.digits,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        block_size=args.block_size,
        train_steps=args.train_steps,
    )
    return pipeline.evaluate_seed(
        pb=pb,
        ebt=ebt,
        ar=ar,
        eval_items=eval_items,
        seed=seed,
        checkpoint=checkpoint,
        checkpoint_reused=reused,
        training_diverged=diverged,
        device=device,
        digits=args.digits,
        n_eval=args.n_eval,
        beam=args.beam,
        topk=args.topk,
    )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:  # pragma: no cover - live GPU path.
    import torch

    started = time.monotonic()
    preconditions = check_preconditions()
    model_specs = {
        "dim": args.dim,
        "n_layers": args.layers,
        "n_heads": args.heads,
        "block_size": args.block_size,
        "digits": args.digits,
        "train_steps": args.train_steps,
        "beam": args.beam,
        "topk": args.topk,
        "n_eval": args.n_eval,
        "fresh_seed_policy": "seeds_distinct_from_1_2_3_no_checkpoint_reuse",
        "reused_measurement_pipeline": "experiment_3882_thesis_a_partb_killgate",
        "ar_best_of_n": "matched_to_cumulative_beam_energy_evals_with_exp3727_budget_matcher",
        "ebt": "scaled_ebt_retrained_from_scratch",
        "ar": "matched_scaled_ar_retrained_from_scratch",
    }
    output_path = REPO_ROOT / OUTPUT_REL_PATH
    if not preconditions.cuda:
        artifact = build_blocked_artifact(
            honest_verdict="blocked_no_cuda",
            preconditions=preconditions,
            model_specs=model_specs,
            started_s=started,
            finished_s=time.monotonic(),
        )
        write_artifact(output_path, artifact)
        return artifact
    if not preconditions.scaled_harness_import:
        artifact = build_blocked_artifact(
            honest_verdict="blocked_scaled_harness_import",
            preconditions=preconditions,
            model_specs=model_specs,
            started_s=started,
            finished_s=time.monotonic(),
        )
        write_artifact(output_path, artifact)
        return artifact
    if not preconditions.exp3882_pipeline_import or not preconditions.exp3882_reusable_functions:
        artifact = build_blocked_artifact(
            honest_verdict="blocked_exp3882_reuse",
            preconditions=preconditions,
            model_specs=model_specs,
            started_s=started,
            finished_s=time.monotonic(),
        )
        write_artifact(output_path, artifact)
        return artifact

    pipeline = load_exp3882_pipeline()
    pb, sc = pipeline.load_scaled_modules()
    pipeline.VOCAB = int(pb.VOCAB)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    evaluations: list[Any] = []
    with fresh_checkpoint_policy(pipeline, results_dir=REPO_ROOT / "results"):
        for seed in args.seeds:
            evaluations.append(
                evaluate_replication_seed(
                    pipeline=pipeline,
                    pb=pb,
                    sc=sc,
                    seed=seed,
                    device=device,
                    args=args,
                )
            )
            torch.cuda.empty_cache()

    artifact = build_artifact(
        seed_evaluations=evaluations,
        all_configured_seeds=args.seeds,
        preconditions=preconditions,
        model_specs=model_specs,
        started_s=started,
        finished_s=time.monotonic(),
        inference_substrate="live_llm_inference",
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    write_artifact(output_path, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--digits", type=int, default=DEFAULT_DIGITS)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--n-eval", type=int, default=DEFAULT_N_EVAL)
    parser.add_argument("--beam", type=int, default=DEFAULT_BEAM)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    artifact = run_experiment(parse_args(argv))
    print(artifact["honest_verdict"], flush=True)
    print(f"[artifact] {REPO_ROOT / OUTPUT_REL_PATH}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
