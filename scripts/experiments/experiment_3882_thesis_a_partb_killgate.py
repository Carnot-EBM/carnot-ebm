#!/usr/bin/env python3
"""Exp 3882: Thesis-A part-b kill-gate beam-search rerun.

Spec refs: REQ-EBT-3882, SCENARIO-EBT-3882-IMPORT,
SCENARIO-EBT-3882-SCHEMA, SCENARIO-EBT-3882-ADJUDICATION.

This reruns the headroom-confirmed Thesis-A part-b probe after the Exp 3871
scaled-harness import bug.  It asks whether global discrete beam search over
cumulative EBT energy recovers generation accuracy when AR has confirmed
held-out headroom.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))  # pragma: no cover - depends on caller sys.path.

from carnot.phase3.matched_compute_eval_harness import match_ar_best_of_m  # noqa: E402


EXPERIMENT_ID = 3882
SCHEMA = "carnot.experiment_3882_thesis_a_partb_killgate.v1"
OUTPUT_REL_PATH = Path("results/experiment_3882_thesis_a_partb_killgate.json")
DEFAULT_SEEDS = (1, 2, 3)
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
VOCAB = 258

BARE_REQUIRED_FIELDS = (
    "positive_control_passed",
    "ar_best_accuracy",
    "ebt_argmin_accuracy",
    "ebt_beam_accuracy",
    "adjudication",
    "matched_flops_ratio",
    "seeds_used",
    "n_heldout",
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
    "seed_evaluations",
    "field_principles",
)

FIELD_PRINCIPLES = {
    "positive_control_passed": (
        "BARE BOOL - true only when best held-out AR is in [0.4,0.95] before "
        "any ARTIFACT/FUNDAMENTAL verdict."
    ),
    "ar_best_accuracy": (
        "Positive control value - AR<0.3 means no headroom and INCONCLUSIVE."
    ),
    "ebt_argmin_accuracy": "Greedy energy-argmin decode accuracy.",
    "ebt_beam_accuracy": "Global cumulative-energy discrete beam-search accuracy.",
    "adjudication": "ARTIFACT | FUNDAMENTAL | INCONCLUSIVE.",
    "matched_flops_ratio": "EBT beam / AR best-of-N inference-FLOP ratio.",
    "seeds_used": "Random seeds actually evaluated.",
    "n_heldout": "Held-out examples evaluated for the selected seed.",
    "preconditions_checked": "CUDA and corrected scaled-harness import evidence.",
    "model_specs": "Scaled model and decode configuration.",
    "random_seed": "Selected seed used for the kill-gate decision.",
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
    python_executable: str = field(default_factory=lambda: sys.executable)
    cuda_devices: list[str] = field(default_factory=list)
    scaled_harness_error: str | None = None


@dataclass(frozen=True)
class SeedEvaluation:
    """One seed's headroom, EBT decode metrics, and matched-compute accounting."""

    seed: int
    checkpoint_path: str
    checkpoint_reused: bool
    n_heldout: int
    ar_best_accuracy: float
    ar_best_of_n: int
    ar_forward_evals: int
    ebt_argmin_accuracy: float | None
    ebt_argmin_evals: int | None
    ebt_beam_accuracy: float | None
    ebt_beam_evals: int | None
    matched_flops_ratio: float | None
    samples: list[dict[str, Any]]
    training_diverged: bool


@dataclass(frozen=True)
class AdjudicationResult:
    """Threshold outcome for the Exp 3882 falsification gate."""

    positive_control_passed: bool
    adjudication: str
    honest_verdict: str


def load_scaled_modules() -> tuple[Any, Any]:  # pragma: no cover - covered by live preflight.
    """Import the scaled Thesis-A harness through the fixed scripts path."""

    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    scaled = importlib.import_module("thesis_a_part_b_scaled")
    return scaled.pb, scaled


def adjudicate(ar_best: float, ebt_argmin: float | None, ebt_beam: float | None) -> AdjudicationResult:
    """Apply SCENARIO-EBT-3882 thresholds without touching model code."""

    positive_control_passed = 0.4 <= ar_best <= 0.95
    if not positive_control_passed:
        return AdjudicationResult(
            positive_control_passed=False,
            adjudication="INCONCLUSIVE",
            honest_verdict=f"complete: thesis_a_partb_INCONCLUSIVE_no_headroom_ar{ar_best:.3f}",
        )

    argmin = float(ebt_argmin if ebt_argmin is not None else 0.0)
    beam = float(ebt_beam if ebt_beam is not None else 0.0)
    if beam >= 0.5 * ar_best:
        return AdjudicationResult(
            positive_control_passed=True,
            adjudication="ARTIFACT",
            honest_verdict=(
                f"complete: thesis_a_partb_ARTIFACT_beam{beam:.3f}_"
                f"recovers_vs_argmin{argmin:.3f}_ar{ar_best:.3f}_"
                "greedy_was_bottleneck"
            ),
        )
    if beam < 0.2 * ar_best and argmin < 0.2 * ar_best:
        return AdjudicationResult(
            positive_control_passed=True,
            adjudication="FUNDAMENTAL",
            honest_verdict=(
                f"complete: thesis_a_partb_FUNDAMENTAL_beam{beam:.3f}_"
                f"argmin{argmin:.3f}_both_fail_vs_ar{ar_best:.3f}_"
                "landscape_misshaped"
            ),
        )
    return AdjudicationResult(
        positive_control_passed=True,
        adjudication="INCONCLUSIVE",
        honest_verdict=(
            f"complete: thesis_a_partb_INCONCLUSIVE_beam{beam:.3f}_"
            f"argmin{argmin:.3f}_ar{ar_best:.3f}_thresholds_not_decisive"
        ),
    )


def matched_ar_best_of_n(*, n_heldout: int, ans_len: int, beam: int) -> int:
    """Select the AR best-of-N budget matching cumulative beam energy evals."""

    beam_eval_total = VOCAB * (1 + beam * max(0, ans_len - 1)) * n_heldout
    ar_single_total = ans_len * n_heldout
    return match_ar_best_of_m(
        target_total_flops=beam_eval_total,
        ar_single_sample_total_flops=ar_single_total,
        tolerance=0.01,
    ).ar_best_of_m


@torch.no_grad()
def ebt_beam_generate(ebt, pid, ans_len, device, beam=8, topk=12):
    """GLOBAL discrete search: beam search over the answer tokens minimising cumulative
    per-position EBT energy, evaluated only at valid token embeddings. Returns
    (best_ids, n_energy_evals). Cost = beam * VOCAB per generated token."""
    emb = ebt.token_embedding.weight
    cand_ids = torch.arange(VOCAB, device=device)
    cand_emb = emb[cand_ids]
    beams = [(list(pid), 0.0)]
    nf = 0
    for _ in range(ans_len):
        expanded = []
        for ids, cum in beams:
            ctx = torch.tensor([ids], device=device); m = ctx.shape[1]
            orig = ebt.token_embedding(ctx).expand(VOCAB, -1, -1)
            known = ebt.token_embedding(ctx[:, 1:]).expand(VOCAB, -1, -1) if m >= 2 \
                else torch.zeros((VOCAB, 0, emb.shape[1]), device=device)
            pred = torch.cat([known, cand_emb.unsqueeze(1)], dim=1)
            e = ebt(orig, pred)[:, -1, 0]
            nf += VOCAB
            low_e, idx = torch.topk(e, topk, largest=False)
            for j in range(topk):
                expanded.append((ids + [int(cand_ids[int(idx[j])])], cum + float(low_e[j])))
        expanded.sort(key=lambda x: x[1])
        beams = expanded[:beam]
    return beams[0][0][len(pid):], nf


def _stable_checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _select_seed(evaluations: Sequence[SeedEvaluation]) -> SeedEvaluation:
    if not evaluations:
        raise ValueError("at least one seed evaluation is required")
    passing = [item for item in evaluations if 0.4 <= item.ar_best_accuracy <= 0.95]
    if passing:
        return max(passing, key=lambda item: (item.ar_best_accuracy, -item.seed))
    return max(evaluations, key=lambda item: (item.ar_best_accuracy, -item.seed))


def build_artifact(
    *,
    seed_evaluations: Sequence[SeedEvaluation],
    all_configured_seeds: Sequence[int],
    preconditions: PreconditionReport,
    model_specs: dict[str, Any],
    started_s: float,
    finished_s: float,
    inference_substrate: str,
) -> dict[str, Any]:
    """Build the terminal JSON artifact with bare required fields."""

    evaluations = list(seed_evaluations)
    selected = _select_seed(evaluations)
    gate = adjudicate(
        selected.ar_best_accuracy,
        selected.ebt_argmin_accuracy,
        selected.ebt_beam_accuracy,
    )
    seeds_used = [item.seed for item in evaluations]
    checksum_payload = {
        "experiment": EXPERIMENT_ID,
        "selected_seed": selected.seed,
        "configured_seeds": list(all_configured_seeds),
        "seed_evaluations": [asdict(item) for item in evaluations],
        "model_specs": model_specs,
        "preconditions": asdict(preconditions),
        "gate": asdict(gate),
    }
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "honest_verdict": gate.honest_verdict,
        "positive_control_passed": bool(gate.positive_control_passed),
        "ar_best_accuracy": selected.ar_best_accuracy,
        "ebt_argmin_accuracy": selected.ebt_argmin_accuracy,
        "ebt_beam_accuracy": selected.ebt_beam_accuracy,
        "adjudication": gate.adjudication,
        "matched_flops_ratio": selected.matched_flops_ratio,
        "seeds_used": seeds_used,
        "n_heldout": selected.n_heldout,
        "preconditions_checked": asdict(preconditions),
        "model_specs": dict(model_specs, selected_seed=selected.seed),
        "random_seed": selected.seed,
        "random_seeds_used": seeds_used,
        "reproducibility_checksum": _stable_checksum(checksum_payload),
        "duration_s": round(float(finished_s - started_s), 2),
        "inference_substrate": inference_substrate,
        "all_configured_seeds": list(all_configured_seeds),
        "seed_evaluations": [asdict(item) for item in evaluations],
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
        "positive_control_passed": False,
        "ar_best_accuracy": None,
        "ebt_argmin_accuracy": None,
        "ebt_beam_accuracy": None,
        "adjudication": "INCONCLUSIVE",
        "matched_flops_ratio": None,
        "seeds_used": [],
        "n_heldout": 0,
        "preconditions_checked": asdict(preconditions),
        "model_specs": dict(model_specs),
        "random_seed": None,
        "random_seeds_used": [],
        "reproducibility_checksum": _stable_checksum(checksum_payload),
        "duration_s": round(float(finished_s - started_s), 2),
        "inference_substrate": "blocked_precondition",
        "all_configured_seeds": [],
        "seed_evaluations": [],
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def validate_artifact(artifact: dict[str, Any]) -> list[str]:
    """Return schema errors that make the Exp 3882 artifact non-terminal."""

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
    if not isinstance(artifact.get("positive_control_passed"), bool):
        errors.append("positive_control_passed must be a bare bool")

    for metric in ("ar_best_accuracy", "ebt_argmin_accuracy", "ebt_beam_accuracy", "matched_flops_ratio"):
        value = artifact.get(metric)
        if value is not None and not isinstance(value, (int, float)):
            errors.append(f"{metric} must be numeric or null")
    if artifact.get("adjudication") not in {"ARTIFACT", "FUNDAMENTAL", "INCONCLUSIVE"}:
        errors.append("adjudication must be ARTIFACT, FUNDAMENTAL, or INCONCLUSIVE")
    if not isinstance(artifact.get("seeds_used"), list):
        errors.append("seeds_used must be a list")
    if not isinstance(artifact.get("random_seeds_used"), list):
        errors.append("random_seeds_used must be a list")
    if not isinstance(artifact.get("n_heldout"), int) or isinstance(artifact.get("n_heldout"), bool):
        errors.append("n_heldout must be an integer")
    if not isinstance(artifact.get("preconditions_checked"), dict):
        errors.append("preconditions_checked must be an object")
    if not isinstance(artifact.get("model_specs"), dict):
        errors.append("model_specs must be an object")
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
    """Check CUDA and the corrected scaled-harness import before live work."""

    cuda = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    device_names = [torch.cuda.get_device_name(i) for i in range(device_count)] if cuda else []
    try:
        load_scaled_modules()
        scaled_import = True
        error = None
    except Exception as exc:
        scaled_import = False
        error = repr(exc)
    return PreconditionReport(
        cuda=cuda,
        cuda_device_count=device_count,
        cuda_devices=device_names,
        scaled_harness_import=scaled_import,
        scaled_harness_error=error,
    )


def _set_seed(seed: int) -> None:  # pragma: no cover - live model path.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint_path(seed: int) -> Path:  # pragma: no cover - live model path.
    return REPO_ROOT / "results" / f"experiment_3882_thesis_a_partb_seed{seed}.pt"


def _scaled_artifact_checkpoint(seed: int) -> Path | None:  # pragma: no cover - live model path.
    artifact_path = REPO_ROOT / "results" / f"thesis_a_part_b_scaled_seed{seed}.json"
    if not artifact_path.exists():
        return None
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    candidates: list[Any] = [
        artifact.get("checkpoint_path"),
        artifact.get("model_checkpoint_path"),
        artifact.get("trained_checkpoint_path"),
    ]
    checkpoint = artifact.get("checkpoint")
    if isinstance(checkpoint, dict):
        candidates.extend([checkpoint.get("path"), checkpoint.get("checkpoint_path")])
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.exists():
            return path
    return None


def _build_data(pb: Any, *, digits: int, seed: int, block_size: int) -> tuple[Any, list[tuple[str, str]]]:  # pragma: no cover
    mu = (10 ** digits) ** 2
    n_train = min(40_000, int(mu * 0.7))
    train_items = pb.build_corpus(digits, n_train, seed)
    train_prompts = {item[0] for item in train_items}
    eval_items = pb.build_corpus(digits, 4_000, seed + 777, exclude=train_prompts)
    blocks = pb.corpus_to_blocks(train_items, block_size)
    return blocks, eval_items


def _load_or_train_models(  # pragma: no cover - live GPU path.
    *,
    pb: Any,
    sc: Any,
    seed: int,
    device: torch.device,
    digits: int,
    dim: int,
    layers: int,
    heads: int,
    block_size: int,
    train_steps: int,
) -> tuple[Any, Any, list[tuple[str, str]], Path, bool, bool]:
    _set_seed(seed)
    blocks, eval_items = _build_data(pb, digits=digits, seed=seed, block_size=block_size)
    ebt, ar = sc.build_models(dim, layers, heads, block_size, device)
    checkpoint = _scaled_artifact_checkpoint(seed) or _checkpoint_path(seed)
    reused = False
    training_diverged = False
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        ebt.load_state_dict(state["ebt"])
        ar.load_state_dict(state["ar"])
        training_diverged = bool(state.get("nan", False))
        reused = True
        print(f"[resume] seed={seed} loaded {checkpoint} nan={training_diverged}", flush=True)
    else:
        print(f"[train] seed={seed} steps={train_steps} checkpoint={checkpoint}", flush=True)
        training_diverged = bool(
            pb.train_models(
                ebt,
                ar,
                blocks,
                device,
                train_steps,
                bs=16,
                langevin=(5, 15),
                log=lambda message: print(f"[seed {seed}] {message}", flush=True),
            )
        )
        torch.save(
            {
                "ebt": ebt.state_dict(),
                "ar": ar.state_dict(),
                "nan": training_diverged,
                "config": {
                    "seed": seed,
                    "digits": digits,
                    "dim": dim,
                    "layers": layers,
                    "heads": heads,
                    "block_size": block_size,
                    "train_steps": train_steps,
                },
            },
            checkpoint,
        )
        print(f"[ckpt] seed={seed} saved {checkpoint} nan={training_diverged}", flush=True)
    return ebt, ar, eval_items, checkpoint, reused, training_diverged


def evaluate_seed(  # pragma: no cover - live GPU path.
    *,
    pb: Any,
    ebt: Any,
    ar: Any,
    eval_items: Sequence[tuple[str, str]],
    seed: int,
    checkpoint: Path,
    checkpoint_reused: bool,
    training_diverged: bool,
    device: torch.device,
    digits: int,
    n_eval: int,
    beam: int,
    topk: int,
) -> SeedEvaluation:
    """Decode held-out examples with AR first, then EBT only if headroom holds."""

    ans_len = digits + 1
    items = list(eval_items[:n_eval])
    if not items:
        raise ValueError("held-out eval set is empty")
    ar_best_of_n = matched_ar_best_of_n(n_heldout=len(items), ans_len=ans_len, beam=beam)
    _set_seed(seed + 100_000)
    ebt.eval()
    ar.eval()
    ar_correct = 0
    ar_forward = 0
    ar_outputs: list[list[int]] = []
    for index, (prompt, true_answer) in enumerate(items):
        pid = pb.enc(prompt)
        true = pb.enc(true_answer)
        ar_ids, nf = pb.ar_selfconsistency(ar, pid, ans_len, device, ar_best_of_n)
        ar_forward += int(nf)
        ar_correct += int(ar_ids == true)
        ar_outputs.append(ar_ids)
        if (index + 1) % 20 == 0:
            print(f"[ar seed={seed}] {index + 1}/{len(items)} arN={ar_correct / (index + 1):.3f}", flush=True)

    n = len(items)
    ar_accuracy = ar_correct / n
    if not (0.4 <= ar_accuracy <= 0.95):
        return SeedEvaluation(
            seed=seed,
            checkpoint_path=str(checkpoint.relative_to(REPO_ROOT) if checkpoint.is_absolute() else checkpoint),
            checkpoint_reused=checkpoint_reused,
            n_heldout=n,
            ar_best_accuracy=ar_accuracy,
            ar_best_of_n=ar_best_of_n,
            ar_forward_evals=ar_forward,
            ebt_argmin_accuracy=None,
            ebt_argmin_evals=None,
            ebt_beam_accuracy=None,
            ebt_beam_evals=None,
            matched_flops_ratio=None,
            samples=[],
            training_diverged=training_diverged,
        )

    argmin_correct = 0
    beam_correct = 0
    argmin_evals = 0
    beam_evals = 0
    samples: list[dict[str, Any]] = []
    for index, (prompt, true_answer) in enumerate(items):
        pid = pb.enc(prompt)
        true = pb.enc(true_answer)
        argmin_ids, nf = pb.ebt_generate(ebt, pid, ans_len, device)
        argmin_evals += int(nf)
        argmin_correct += int(argmin_ids == true)
        beam_ids, nf = ebt_beam_generate(ebt, pid, ans_len, device, beam=beam, topk=topk)
        beam_evals += int(nf)
        beam_correct += int(beam_ids == true)
        if index < 8:
            samples.append(
                {
                    "prompt": prompt,
                    "true": true_answer,
                    "ar_best_of_n": pb.dec_ids(ar_outputs[index]),
                    "ebt_argmin": pb.dec_ids(argmin_ids),
                    "ebt_beam": pb.dec_ids(beam_ids),
                }
            )
        if (index + 1) % 20 == 0:
            denom = index + 1
            print(
                f"[ebt seed={seed}] {denom}/{len(items)} "
                f"argmin={argmin_correct / denom:.3f} "
                f"beam={beam_correct / denom:.3f}",
                flush=True,
            )
    return SeedEvaluation(
        seed=seed,
        checkpoint_path=str(checkpoint.relative_to(REPO_ROOT) if checkpoint.is_absolute() else checkpoint),
        checkpoint_reused=checkpoint_reused,
        n_heldout=n,
        ar_best_accuracy=ar_accuracy,
        ar_best_of_n=ar_best_of_n,
        ar_forward_evals=ar_forward,
        ebt_argmin_accuracy=argmin_correct / n,
        ebt_argmin_evals=argmin_evals,
        ebt_beam_accuracy=beam_correct / n,
        ebt_beam_evals=beam_evals,
        matched_flops_ratio=(beam_evals / ar_forward) if ar_forward else None,
        samples=samples,
        training_diverged=training_diverged,
    )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:  # pragma: no cover - live GPU path.
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
        "ar_best_of_n": "matched_to_cumulative_beam_energy_evals_with_exp3727_budget_matcher",
        "ebt": "scaled_ebt_from_scratch_or_checkpoint",
        "ar": "matched_scaled_ar_from_scratch_or_checkpoint",
    }
    if not preconditions.cuda:
        artifact = build_blocked_artifact(
            honest_verdict="blocked_no_cuda",
            preconditions=preconditions,
            model_specs=model_specs,
            started_s=started,
            finished_s=time.monotonic(),
        )
        write_artifact(REPO_ROOT / OUTPUT_REL_PATH, artifact)
        return artifact
    if not preconditions.scaled_harness_import:
        artifact = build_blocked_artifact(
            honest_verdict="blocked_scaled_harness_import",
            preconditions=preconditions,
            model_specs=model_specs,
            started_s=started,
            finished_s=time.monotonic(),
        )
        write_artifact(REPO_ROOT / OUTPUT_REL_PATH, artifact)
        return artifact

    pb, sc = load_scaled_modules()
    global VOCAB
    VOCAB = int(pb.VOCAB)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    evaluations: list[SeedEvaluation] = []
    for seed in args.seeds:
        ebt, ar, eval_items, checkpoint, reused, diverged = _load_or_train_models(
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
        evaluations.append(
            evaluate_seed(
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
        )
        del ebt, ar
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
    write_artifact(REPO_ROOT / OUTPUT_REL_PATH, artifact)
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
