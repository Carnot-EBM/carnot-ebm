#!/usr/bin/env python3
"""Exp 3883: EBT System-2 K-curve on the Exp 3882 checkpoint.

Spec refs: REQ-EBT-3883, SCENARIO-EBT-3883-UPSTREAM,
SCENARIO-EBT-3883-SCHEMA, SCENARIO-EBT-3883-FALSIFICATION.

This isolates the narrow claim that more inference-time energy-descent steps K
improve prediction. It reuses the Exp 3882 confirmed-headroom checkpoint and
sweeps K without retraining the EBT.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 3883
SCHEMA = "carnot.experiment_3883_ebt_system2_kcurve.v1"
UPSTREAM_REL_PATH = Path("results/experiment_3882_thesis_a_partb_killgate.json")
OUTPUT_REL_PATH = Path("results/experiment_3883_ebt_system2_kcurve.json")
DEFAULT_DEVICE = "cuda:0"
DEFAULT_K_VALUES = (1, 2, 4, 8, 16)
DEFAULT_DECODER_TRAIN_STEPS = 500
DEFAULT_DECODER_TRAIN_K = 16
DEFAULT_DECODER_BATCH_SIZE = 32

BARE_REQUIRED_FIELDS = (
    "accuracy_by_k",
    "k_curve_shape",
    "best_k",
    "best_k_accuracy",
    "n_heldout",
    "seeds_used",
    "preconditions_checked",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
)
REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "honest_verdict",
    *BARE_REQUIRED_FIELDS,
    "field_principles",
)
CURVE_SHAPES = {"MONOTONE_GAIN", "PLATEAU", "DEGRADING"}
FIELD_PRINCIPLES = {
    "accuracy_by_k": "The K-scaling curve - direct evidence for/against 'thinking = energy descent helps'.",
    "k_curve_shape": "MONOTONE_GAIN | PLATEAU | DEGRADING - bounds (or supports) the EBT System-2 claim at this scale.",
    "best_k": "Compute-optimal descent budget; is more inference compute worth it.",
    "best_k_accuracy": "Compute-optimal descent budget; is more inference compute worth it.",
    "n_heldout": "Methodology - real K-sweep inference takes wall-clock (live floor 60s); implausibly short = fabrication.",
    "seeds_used": "Methodology - real K-sweep inference takes wall-clock (live floor 60s); implausibly short = fabrication.",
    "preconditions_checked": "Methodology - real K-sweep inference takes wall-clock (live floor 60s); implausibly short = fabrication.",
    "model_specs": "Methodology - real K-sweep inference takes wall-clock (live floor 60s); implausibly short = fabrication.",
    "random_seed": "Methodology - real K-sweep inference takes wall-clock (live floor 60s); implausibly short = fabrication.",
    "reproducibility_checksum": "Methodology - real K-sweep inference takes wall-clock (live floor 60s); implausibly short = fabrication.",
    "duration_s": "Methodology - real K-sweep inference takes wall-clock (live floor 60s); implausibly short = fabrication.",
    "inference_substrate": "Methodology - real K-sweep inference takes wall-clock (live floor 60s); implausibly short = fabrication.",
}


@dataclass(frozen=True)
class PreconditionReport:
    """Preconditions that decide whether live K-sweep work may run."""

    cuda: bool
    cuda_device_count: int
    scaled_harness_import: bool
    upstream_positive_control: bool | None = None
    checkpoint_loaded: bool | None = None
    python_executable: str = sys.executable
    cuda_devices: list[str] | None = None
    scaled_harness_error: str | None = None
    upstream_error: str | None = None


@dataclass(frozen=True)
class UpstreamContext:
    """Resolved Exp 3882 checkpoint and decode-path provenance."""

    upstream: dict[str, Any]
    selected_seed: int
    checkpoint_path: Path
    checkpoint_state: dict[str, Any]
    decode_path: str
    n_heldout: int


@dataclass(frozen=True)
class KCurveSummary:
    """A compact summary of the K-scaling curve."""

    k_curve_shape: str
    best_k: int
    best_k_accuracy: float
    linear_slope: float


def load_scaled_harness() -> tuple[Any, Any]:  # pragma: no cover - live import path.
    """Import the scaled harness through `scripts/`, not as a package module."""

    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    scaled = importlib.import_module("thesis_a_part_b_scaled")
    return scaled.pb, scaled


def _stable_checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def select_decode_path(upstream: Mapping[str, Any]) -> str:
    """Return the stronger Exp 3882 discrete decode path, breaking ties to beam."""

    beam = _as_float(upstream.get("ebt_beam_accuracy"))
    argmin = _as_float(upstream.get("ebt_argmin_accuracy"))
    return "beam" if beam >= argmin else "argmin"


def _selected_seed(upstream: Mapping[str, Any]) -> int | None:
    for candidate in (
        upstream.get("random_seed"),
        (upstream.get("model_specs") or {}).get("selected_seed")
        if isinstance(upstream.get("model_specs"), dict)
        else None,
    ):
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    return None


def _checkpoint_for_seed(upstream: Mapping[str, Any], selected_seed: int) -> tuple[str | None, int]:
    evaluations = upstream.get("seed_evaluations")
    if not isinstance(evaluations, list):
        return None, int(upstream.get("n_heldout") or 0)
    for item in evaluations:
        if not isinstance(item, dict) or item.get("seed") != selected_seed:
            continue
        checkpoint = item.get("checkpoint_path")
        n_heldout = int(item.get("n_heldout") or upstream.get("n_heldout") or 0)
        return checkpoint if isinstance(checkpoint, str) and checkpoint else None, n_heldout
    return None, int(upstream.get("n_heldout") or 0)


def load_upstream_context(upstream_path: Path, repo_root: Path) -> tuple[UpstreamContext | None, str | None]:
    """Load Exp 3882 and its selected checkpoint, or return a blocking reason."""

    if not upstream_path.exists():
        return None, "experiment_3882 artifact missing"
    try:
        upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "experiment_3882 artifact unreadable"
    if upstream.get("positive_control_passed") is not True:
        return None, "positive_control_passed was not true"
    selected_seed = _selected_seed(upstream)
    if selected_seed is None:
        return None, "selected seed missing"
    checkpoint_raw, n_heldout = _checkpoint_for_seed(upstream, selected_seed)
    if checkpoint_raw is None:
        return None, "selected checkpoint path missing"

    checkpoint_path = Path(checkpoint_raw)
    if not checkpoint_path.is_absolute():
        checkpoint_path = repo_root / checkpoint_path
    try:
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return None, "checkpoint did not load"
    if not isinstance(checkpoint_state, dict) or "ebt" not in checkpoint_state:
        return None, "checkpoint did not load"
    return (
        UpstreamContext(
            upstream=upstream,
            selected_seed=selected_seed,
            checkpoint_path=checkpoint_path,
            checkpoint_state=checkpoint_state,
            decode_path=select_decode_path(upstream),
            n_heldout=n_heldout,
        ),
        None,
    )


def summarize_k_curve(accuracy_by_k: Mapping[int | str, float]) -> KCurveSummary:
    """Classify the K curve for the System-2 falsification gate."""

    normalized = {int(k): float(v) for k, v in accuracy_by_k.items()}
    missing = [k for k in DEFAULT_K_VALUES if k not in normalized]
    if missing:
        raise ValueError(f"missing K values: {missing}")

    values = [normalized[k] for k in DEFAULT_K_VALUES]
    best_k = DEFAULT_K_VALUES[0]
    best_accuracy = values[0]
    for k, accuracy in zip(DEFAULT_K_VALUES[1:], values[1:]):
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy

    eps = 1e-12
    nondecreasing = all(values[index + 1] >= values[index] - eps for index in range(len(values) - 1))
    if nondecreasing and values[-1] > values[0] + eps:
        shape = "MONOTONE_GAIN"
    elif values[-1] < values[0] - eps:
        shape = "DEGRADING"
    else:
        shape = "PLATEAU"

    mean_k = sum(DEFAULT_K_VALUES) / len(DEFAULT_K_VALUES)
    mean_acc = sum(values) / len(values)
    denom = sum((k - mean_k) ** 2 for k in DEFAULT_K_VALUES)
    slope = sum((k - mean_k) * (acc - mean_acc) for k, acc in zip(DEFAULT_K_VALUES, values)) / denom
    if abs(slope) < eps:
        slope = 0.0
    return KCurveSummary(
        k_curve_shape=shape,
        best_k=int(best_k),
        best_k_accuracy=float(best_accuracy),
        linear_slope=float(slope),
    )


def _verdict_for_curve(summary: KCurveSummary, accuracy_by_k: Mapping[int | str, float]) -> str:
    normalized = {int(k): float(v) for k, v in accuracy_by_k.items()}
    if summary.k_curve_shape == "MONOTONE_GAIN" and normalized[16] > normalized[1]:
        return (
            "complete: ebt_system2_SUPPORTED_acc_rises_with_k_"
            f"best_k{summary.best_k}_acc{summary.best_k_accuracy:.3f}"
        )
    return f"complete: ebt_system2_BOUNDED_{summary.k_curve_shape}_no_usable_descent_signal_at_scale"


def build_artifact(
    *,
    accuracy_by_k: Mapping[int | str, float],
    n_heldout: int,
    seeds_used: Sequence[int],
    preconditions: PreconditionReport,
    model_specs: dict[str, Any],
    random_seed: int,
    started_s: float,
    finished_s: float,
    inference_substrate: str,
    samples: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a complete terminal artifact with bare required fields."""

    normalized = {str(k): float(accuracy_by_k[k]) for k in DEFAULT_K_VALUES}
    summary = summarize_k_curve(normalized)
    checksum_payload = {
        "experiment": EXPERIMENT_ID,
        "accuracy_by_k": normalized,
        "summary": asdict(summary),
        "n_heldout": n_heldout,
        "seeds_used": list(seeds_used),
        "model_specs": model_specs,
        "random_seed": random_seed,
        "preconditions": asdict(preconditions),
    }
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "honest_verdict": _verdict_for_curve(summary, normalized),
        "accuracy_by_k": normalized,
        "k_curve_shape": summary.k_curve_shape,
        "best_k": summary.best_k,
        "best_k_accuracy": summary.best_k_accuracy,
        "n_heldout": int(n_heldout),
        "seeds_used": list(seeds_used),
        "preconditions_checked": asdict(preconditions),
        "model_specs": dict(model_specs),
        "random_seed": int(random_seed),
        "reproducibility_checksum": _stable_checksum(checksum_payload),
        "duration_s": round(float(finished_s - started_s), 2),
        "inference_substrate": inference_substrate,
        "k_curve_fit": {"linear_slope_accuracy_per_k": summary.linear_slope},
        "samples": list(samples or []),
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
    """Build a terminal blocked artifact without fabricating K metrics."""

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
        "accuracy_by_k": {},
        "k_curve_shape": None,
        "best_k": None,
        "best_k_accuracy": None,
        "n_heldout": 0,
        "seeds_used": [],
        "preconditions_checked": asdict(preconditions),
        "model_specs": dict(model_specs),
        "random_seed": None,
        "reproducibility_checksum": _stable_checksum(checksum_payload),
        "duration_s": round(float(finished_s - started_s), 2),
        "inference_substrate": "blocked_precondition",
        "k_curve_fit": {"linear_slope_accuracy_per_k": None},
        "samples": [],
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def validate_artifact(artifact: dict[str, Any]) -> list[str]:
    """Return schema errors that make the Exp 3883 artifact non-terminal."""

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

    accuracy_by_k = artifact.get("accuracy_by_k")
    if not isinstance(accuracy_by_k, dict):
        errors.append("accuracy_by_k must be an object")
    elif not ({"value", "principle"} <= set(accuracy_by_k)):
        for key, value in accuracy_by_k.items():
            if not isinstance(key, str):
                errors.append("accuracy_by_k keys must be strings")
                break
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append("accuracy_by_k values must be numeric")
                break

    if artifact.get("k_curve_shape") not in CURVE_SHAPES | {None}:
        errors.append("k_curve_shape must be MONOTONE_GAIN, PLATEAU, DEGRADING, or null")
    if artifact.get("best_k") is not None and (
        not isinstance(artifact.get("best_k"), int) or isinstance(artifact.get("best_k"), bool)
    ):
        errors.append("best_k must be an integer or null")
    if artifact.get("best_k_accuracy") is not None and (
        not isinstance(artifact.get("best_k_accuracy"), (int, float))
        or isinstance(artifact.get("best_k_accuracy"), bool)
    ):
        errors.append("best_k_accuracy must be numeric or null")
    if not isinstance(artifact.get("n_heldout"), int) or isinstance(artifact.get("n_heldout"), bool):
        errors.append("n_heldout must be an integer")
    if not isinstance(artifact.get("seeds_used"), list):
        errors.append("seeds_used must be a list")
    if not isinstance(artifact.get("preconditions_checked"), dict):
        errors.append("preconditions_checked must be an object")
    if not isinstance(artifact.get("model_specs"), dict):
        errors.append("model_specs must be an object")
    if artifact.get("random_seed") is not None and (
        not isinstance(artifact.get("random_seed"), int) or isinstance(artifact.get("random_seed"), bool)
    ):
        errors.append("random_seed must be an integer or null")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum):
        errors.append("reproducibility_checksum must be a sha256 hex string")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration < 0:
        errors.append("duration_s must be non-negative")
    if not isinstance(artifact.get("inference_substrate"), str):
        errors.append("inference_substrate must be a string")
    return errors


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Write sorted JSON so reruns are diffable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_cuda_precondition() -> PreconditionReport:  # pragma: no cover - live preflight.
    cuda = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    device_names = [torch.cuda.get_device_name(index) for index in range(device_count)] if cuda else []
    return PreconditionReport(
        cuda=cuda,
        cuda_device_count=device_count,
        scaled_harness_import=False,
        cuda_devices=device_names,
    )


def _with_precondition(base: PreconditionReport, **updates: Any) -> PreconditionReport:
    values = asdict(base)
    values.update(updates)
    return PreconditionReport(**values)


def _set_seed(seed: int) -> None:  # pragma: no cover - live GPU path.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_data(pb: Any, *, digits: int, seed: int, block_size: int) -> tuple[Any, list[tuple[str, str]]]:  # pragma: no cover
    mu = (10**digits) ** 2
    n_train = min(40_000, int(mu * 0.7))
    train_items = pb.build_corpus(digits, n_train, seed)
    train_prompts = {item[0] for item in train_items}
    eval_items = pb.build_corpus(digits, 4_000, seed + 777, exclude=train_prompts)
    blocks = pb.corpus_to_blocks(train_items, block_size)
    return blocks, eval_items


def _model_specs_from_upstream(context: UpstreamContext, args: argparse.Namespace) -> dict[str, Any]:  # pragma: no cover
    upstream_specs = context.upstream.get("model_specs") if isinstance(context.upstream.get("model_specs"), dict) else {}
    dim = int(upstream_specs.get("dim", 768))
    layers = int(upstream_specs.get("n_layers", upstream_specs.get("layers", 4)))
    heads = int(upstream_specs.get("n_heads", upstream_specs.get("heads", 12)))
    block_size = int(upstream_specs.get("block_size", 48))
    digits = int(upstream_specs.get("digits", 3))
    n_eval = int(upstream_specs.get("n_eval", context.n_heldout or 100))
    return {
        "dim": dim,
        "n_layers": layers,
        "n_heads": heads,
        "block_size": block_size,
        "digits": digits,
        "n_eval": n_eval,
        "k_values": list(DEFAULT_K_VALUES),
        "selected_seed": context.selected_seed,
        "checkpoint_path": str(context.checkpoint_path.relative_to(REPO_ROOT)),
        "upstream_artifact": str(UPSTREAM_REL_PATH),
        "exp3882_stronger_decode_path": context.decode_path,
        "continuous_decoder": "learned emb->VOCAB MLP from thesis_a_part_b_scaled.fit_decoder",
        "decoder_train_k": args.decoder_train_k,
        "decoder_train_steps": args.decoder_train_steps,
        "decoder_batch_size": args.decoder_batch_size,
        "ebt_checkpoint": "exp3882 selected-seed EBT weights, not retrained",
    }


def _load_ebt_and_decoder(  # pragma: no cover - live GPU path.
    *,
    pb: Any,
    scaled: Any,
    context: UpstreamContext,
    model_specs: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[Any, Any, list[tuple[str, str]]]:
    _set_seed(context.selected_seed)
    blocks, eval_items = _build_data(
        pb,
        digits=model_specs["digits"],
        seed=context.selected_seed,
        block_size=model_specs["block_size"],
    )
    ebt, ar = scaled.build_models(
        model_specs["dim"],
        model_specs["n_layers"],
        model_specs["n_heads"],
        model_specs["block_size"],
        device,
    )
    ebt.load_state_dict(context.checkpoint_state["ebt"])
    ebt.eval()
    del ar
    _set_seed(context.selected_seed + 200_000)
    decoder = scaled.fit_decoder(
        ebt,
        blocks,
        device,
        K=args.decoder_train_k,
        steps=args.decoder_train_steps,
        bs=args.decoder_batch_size,
        log=lambda message: print(message, flush=True),
    )
    decoder.eval()
    return ebt, decoder, eval_items


def _evaluate_k_curve(  # pragma: no cover - live GPU path.
    *,
    pb: Any,
    scaled: Any,
    ebt: Any,
    decoder: Any,
    eval_items: Sequence[tuple[str, str]],
    device: torch.device,
    digits: int,
    n_eval: int,
    random_seed: int,
) -> tuple[dict[int, float], list[dict[str, Any]]]:
    ans_len = digits + 1
    items = list(eval_items[:n_eval])
    if not items:
        raise ValueError("held-out eval set is empty")
    accuracy_by_k: dict[int, float] = {}
    sample_rows: dict[int, dict[str, Any]] = {}
    ebt.eval()
    decoder.eval()
    for K in DEFAULT_K_VALUES:
        correct = 0
        for index, (prompt, true_answer) in enumerate(items):
            pid = pb.enc(prompt)
            true_ids = pb.enc(true_answer)
            _set_seed(random_seed + 1_000_000 + index)
            pred_ids, _ = scaled.ebt_descent_generate(ebt, decoder, pid, ans_len, device, K)
            correct += int(pred_ids == true_ids)
            if index < 8:
                row = sample_rows.setdefault(index, {"prompt": prompt, "true": true_answer})
                row[f"k{K}"] = pb.dec_ids(pred_ids)
        accuracy = correct / len(items)
        accuracy_by_k[K] = accuracy
        print(f"[kcurve] K={K} accuracy={accuracy:.3f} n={len(items)}", flush=True)
    return accuracy_by_k, [sample_rows[index] for index in sorted(sample_rows)]


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:  # pragma: no cover - live GPU path.
    started = time.monotonic()
    preconditions = check_cuda_precondition()
    if not preconditions.cuda:
        artifact = build_blocked_artifact(
            honest_verdict="blocked_no_cuda",
            preconditions=preconditions,
            model_specs={"upstream_artifact": str(UPSTREAM_REL_PATH)},
            started_s=started,
            finished_s=time.monotonic(),
        )
        write_artifact(REPO_ROOT / OUTPUT_REL_PATH, artifact)
        return artifact

    context, upstream_error = load_upstream_context(REPO_ROOT / UPSTREAM_REL_PATH, REPO_ROOT)
    if context is None:
        preconditions = _with_precondition(
            preconditions,
            upstream_positive_control=False,
            checkpoint_loaded=False,
            upstream_error=upstream_error,
        )
        artifact = build_blocked_artifact(
            honest_verdict="blocked_upstream_no_headroom",
            preconditions=preconditions,
            model_specs={"upstream_artifact": str(UPSTREAM_REL_PATH)},
            started_s=started,
            finished_s=time.monotonic(),
        )
        write_artifact(REPO_ROOT / OUTPUT_REL_PATH, artifact)
        return artifact
    preconditions = _with_precondition(
        preconditions,
        upstream_positive_control=True,
        checkpoint_loaded=True,
    )

    try:
        pb, scaled = load_scaled_harness()
        preconditions = _with_precondition(preconditions, scaled_harness_import=True, scaled_harness_error=None)
    except Exception as exc:
        preconditions = _with_precondition(
            preconditions,
            scaled_harness_import=False,
            scaled_harness_error=repr(exc),
        )
        artifact = build_blocked_artifact(
            honest_verdict="blocked_scaled_harness_import",
            preconditions=preconditions,
            model_specs={"upstream_artifact": str(UPSTREAM_REL_PATH), "selected_seed": context.selected_seed},
            started_s=started,
            finished_s=time.monotonic(),
        )
        write_artifact(REPO_ROOT / OUTPUT_REL_PATH, artifact)
        return artifact

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model_specs = _model_specs_from_upstream(context, args)
    ebt, decoder, eval_items = _load_ebt_and_decoder(
        pb=pb,
        scaled=scaled,
        context=context,
        model_specs=model_specs,
        device=device,
        args=args,
    )
    accuracy_by_k, samples = _evaluate_k_curve(
        pb=pb,
        scaled=scaled,
        ebt=ebt,
        decoder=decoder,
        eval_items=eval_items,
        device=device,
        digits=model_specs["digits"],
        n_eval=model_specs["n_eval"],
        random_seed=context.selected_seed,
    )
    artifact = build_artifact(
        accuracy_by_k=accuracy_by_k,
        n_heldout=model_specs["n_eval"],
        seeds_used=[context.selected_seed],
        preconditions=preconditions,
        model_specs=model_specs,
        random_seed=context.selected_seed,
        started_s=started,
        finished_s=time.monotonic(),
        inference_substrate="live_llm_inference",
        samples=samples,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    write_artifact(REPO_ROOT / OUTPUT_REL_PATH, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--decoder-train-steps", type=int, default=DEFAULT_DECODER_TRAIN_STEPS)
    parser.add_argument("--decoder-train-k", type=int, default=DEFAULT_DECODER_TRAIN_K)
    parser.add_argument("--decoder-batch-size", type=int, default=DEFAULT_DECODER_BATCH_SIZE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    artifact = run_experiment(parse_args(argv))
    print(artifact["honest_verdict"], flush=True)
    print(f"[artifact] {REPO_ROOT / OUTPUT_REL_PATH}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
