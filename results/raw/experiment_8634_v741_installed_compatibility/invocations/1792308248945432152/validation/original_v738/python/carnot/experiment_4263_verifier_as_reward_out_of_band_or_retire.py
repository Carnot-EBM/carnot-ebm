"""Exp 4263 verifier-as-reward out-of-band package or retirement.

Spec refs: REQ-LEARN-4263, SCENARIO-LEARN-4263-READY,
SCENARIO-LEARN-4263-BLOCKED.

The prior in-window LoRA attempts failed before science because the conductor
window was too small for real training. This module deliberately performs only
the durable preparation step: it freezes the reward-weighted corpus, writes the
operator-run training command, and records whether the axis is ready or must be
retired. Training remains outside this process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4263_verifier_as_reward_out_of_band_or_retire.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_STABLE_CHECKPOINT = (
    REPO_ROOT
    / "results"
    / "verifier_reward_3arm_lora_rft"
    / "code_verifier_reward_lora_rft_a83b52882c198954"
)
DEFAULT_PACKAGE_DIR = REPO_ROOT / "results" / "experiment_4263_verifier_as_reward_out_of_band"
WEIGHTED_CORPUS_FILENAME = "reward_weighted_corpus.jsonl"
RUNNER_FILENAME = "run_out_of_band_reward_weighted_lora.py"
PACKAGE_MANIFEST_FILENAME = "package_manifest.json"
TRAINING_RESULT_FILENAME = "training_result.json"
RANDOM_SEED = 4263
MIN_OPTIMIZER_STEPS = 20
MIN_PLAUSIBLE_DURATION_S = 10.0
TRAINABLE_BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SPEC_REFS = ["REQ-LEARN-4263", "SCENARIO-LEARN-4263-READY", "SCENARIO-LEARN-4263-BLOCKED"]
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "ready_for_out_of_band",
    "verifier_as_reward_retired",
    "out_of_band_runner_path",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An out-of-band package READY and an honest RETIRE are BOTH COMPLETE -- "
        "both resolve the 7x-failed axis without an 8th doomed in-window attempt."
    ),
    "ready_for_out_of_band": (
        "BARE bool: true iff the reward-weighted corpus is precomputed AND a one-command out-of-band "
        "runner + validation harness are written -- the operator then runs the training (TRM pattern)."
    ),
    "verifier_as_reward_retired": (
        "BARE bool: true iff even out-of-band prep is infeasible -> the in-loop axis is retired and "
        "FoVer +0.0185 (exp2837) stands as the self-learning evidence."
    ),
    "out_of_band_runner_path": (
        "Path to the written one-command runner + validation harness the operator executes; the "
        "deliverable that moves training out of the infeasible window."
    ),
    "verifier_is_oracle": (
        "BARE bool=true -- the reward axis uses the verifier label as a training reward (the honest "
        "reward framing); declared per Circularity Discipline."
    ),
    "random_seed": "Determinism precondition for the corpus weighting.",
    "reproducibility_checksum": (
        "Hash of the precomputed reward-weighted corpus; lets the out-of-band run be reproducible."
    ),
    "model_specs": "The small NON-Qwen base + RAFT/RWR weighting + runner config; required methodology.",
}


@dataclass(frozen=True)
class CorpusBundle:
    """Loaded A/B/C corpus state before any training package is written."""

    ready: bool
    rows_by_arm: dict[str, list[dict[str, Any]]]
    corpus_paths: dict[str, str]
    corpus_sizes: dict[str, int]
    missing: list[str]
    error: str | None = None


@dataclass(frozen=True)
class WeightedCorpus:
    """Ready-to-train rows plus the A-vs-B contrast audit result."""

    rows: list[dict[str, Any]]
    corpus_sizes: dict[str, int]
    reward_weight_counts: dict[str, int]
    supports_clean_avsb: bool
    avsb_diagnostic: str


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} contains a non-object JSONL row")
            rows.append(payload)
    return rows


def load_abc_corpora(stable_checkpoint_path: str | Path = DEFAULT_STABLE_CHECKPOINT) -> CorpusBundle:
    """REQ-LEARN-4263: load all three intact corpora before any prep work.

    Missing or empty A/B/C files are terminal for this experiment because a
    synthesized replacement would erase the A-vs-B-vs-C control that makes the
    reward claim interpretable.
    """

    root = Path(stable_checkpoint_path)
    paths = {arm: root / "corpora" / f"arm_{arm}.jsonl" for arm in ("A", "B", "C")}
    missing = [arm for arm, path in paths.items() if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        return CorpusBundle(
            ready=False,
            rows_by_arm={},
            corpus_paths={arm: str(path) for arm, path in paths.items()},
            corpus_sizes={},
            missing=missing,
            error="missing_or_empty_" + "_".join(missing),
        )
    try:
        rows_by_arm = {arm: _load_jsonl(path) for arm, path in paths.items()}
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return CorpusBundle(
            ready=False,
            rows_by_arm={},
            corpus_paths={arm: str(path) for arm, path in paths.items()},
            corpus_sizes={},
            missing=list(paths),
            error=f"{type(exc).__name__}: {exc}",
        )
    empty = [arm for arm, rows in rows_by_arm.items() if not rows]
    if empty:
        return CorpusBundle(
            ready=False,
            rows_by_arm=rows_by_arm,
            corpus_paths={arm: str(path) for arm, path in paths.items()},
            corpus_sizes={arm: len(rows) for arm, rows in rows_by_arm.items()},
            missing=empty,
            error="empty_" + "_".join(empty),
        )
    return CorpusBundle(
        ready=True,
        rows_by_arm=rows_by_arm,
        corpus_paths={arm: str(path) for arm, path in paths.items()},
        corpus_sizes={arm: len(rows) for arm, rows in rows_by_arm.items()},
        missing=[],
    )


def reward_weighting_scheme() -> dict[str, Any]:
    """Return the fixed RAFT/RWR weights used by both corpus and runner."""

    return {
        "method": "offline_raft_reward_weighted_regression_filtered_bc",
        "verifier_certified_weight": 1.0,
        "random_label_control_weight": 0.25,
        "hidden_gold_weight": 1.0,
        "live_generation": False,
        "in_window_training": False,
        "fixed_optimizer_steps": MIN_OPTIMIZER_STEPS,
        "rationale": "Verifier-certified and hidden-gold rows get full BC weight; same-generator random labels are kept only as a low-weight control.",
    }


def _reward_for_arm(arm: str, scheme: Mapping[str, Any]) -> tuple[float, str]:
    if arm == "B":
        return float(scheme["random_label_control_weight"]), "same_generator_random_label_control"
    if arm == "C":
        return float(scheme["hidden_gold_weight"]), "hidden_gold_positive_control"
    return float(scheme["verifier_certified_weight"]), "verifier_certified"


def _row_uid(arm: str, index: int, row: Mapping[str, Any]) -> str:
    payload = {
        "arm": arm,
        "completion": row.get("completion", ""),
        "index": index,
        "prompt": row.get("prompt", ""),
        "task_id": row.get("task_id", ""),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _weighted_row(row: Mapping[str, Any], *, arm: str, index: int, scheme: Mapping[str, Any]) -> dict[str, Any]:
    weight, source = _reward_for_arm(arm, scheme)
    weighted = dict(row)
    weighted.update(
        {
            "arm_id": arm,
            "source_arm": str(row.get("arm") or f"arm_{arm}"),
            "source_index": int(index),
            "reward_weight": weight,
            "reward_source": source,
            "reward_method": str(scheme["method"]),
            "example_uid": _row_uid(arm, index, row),
        }
    )
    return weighted


def _truthy_label(row: Mapping[str, Any], key: str) -> bool | None:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    return None


def _supports_clean_avsb(rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[bool, str]:
    a_rows = list(rows_by_arm.get("A") or [])
    b_rows = list(rows_by_arm.get("B") or [])
    c_rows = list(rows_by_arm.get("C") or [])
    if not a_rows or not b_rows or not c_rows:
        return False, "one_or_more_arms_empty"
    a_has_positive = any(
        _truthy_label(row, "hidden_pass") is True or _truthy_label(row, "visible_perfect") is True
        for row in a_rows
    )
    b_has_negative = any(
        _truthy_label(row, "hidden_pass") is False or _truthy_label(row, "visible_perfect") is False
        for row in b_rows
    )
    b_is_random_label = any("random" in str(row.get("arm") or "").lower() for row in b_rows)
    if not a_has_positive:
        return False, "arm_a_has_no_positive_verifier_certified_rows"
    if not b_has_negative:
        return False, "arm_b_has_no_random_label_failure_contrast"
    if not b_is_random_label:
        return False, "arm_b_not_marked_same_generator_random_label"
    return True, "clean_a_vs_b_supported_by_verifier_certified_vs_random_label_control"


def build_reward_weighted_corpus(
    bundle: CorpusBundle,
    *,
    random_seed: int = RANDOM_SEED,
    scheme: Mapping[str, Any] | None = None,
) -> WeightedCorpus:
    """REQ-LEARN-4263: create deterministic per-example reward weights.

    The row order is interleaved A/B/C rather than globally shuffled so a human
    can inspect the contrast directly. The seed is still recorded in every
    artifact and manifest because future variants may sample or downselect.
    """

    if not bundle.ready:
        return WeightedCorpus([], {}, {}, False, bundle.error or "abc_corpora_missing")
    weights = dict(scheme or reward_weighting_scheme())
    max_rows = max(bundle.corpus_sizes.values())
    rows: list[dict[str, Any]] = []
    for row_index in range(max_rows):
        for arm in ("A", "B", "C"):
            arm_rows = bundle.rows_by_arm[arm]
            if row_index < len(arm_rows):
                weighted = _weighted_row(arm_rows[row_index], arm=arm, index=row_index, scheme=weights)
                weighted["random_seed"] = int(random_seed)
                rows.append(weighted)
    supports, diagnostic = _supports_clean_avsb(bundle.rows_by_arm)
    return WeightedCorpus(
        rows=rows,
        corpus_sizes=dict(bundle.corpus_sizes),
        reward_weight_counts=_reward_weight_counts(rows),
        supports_clean_avsb=supports,
        avsb_diagnostic=diagnostic,
    )


def _reward_weight_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("reward_weight", ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(_jsonable(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode(
        "utf-8"
    )


def checksum_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    return f"sha256:{hashlib.sha256(_jsonl_bytes(rows)).hexdigest()}"


def sha256_file(path: str | Path) -> str:
    return f"sha256:{hashlib.sha256(Path(path).read_bytes()).hexdigest()}"


def write_jsonl_with_checksum(rows: Sequence[Mapping[str, Any]], path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_jsonl_bytes(rows))
    return sha256_file(output)


def _runner_script_text(
    *,
    corpus_path: Path,
    output_path: Path,
    adapter_dir: Path,
    base_model: str,
    random_seed: int,
) -> str:
    """Return the operator-run trainer script with integrated validation."""

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return f'''#!/usr/bin/env python3
"""Out-of-band LoRA trainer and validator for Exp 4263.

Run this outside the conductor window. It loads a small non-Qwen base through
AutoModelForCausalLM, attaches LoRA, trains for real optimizer steps on the
precomputed reward-weighted corpus, and fails if the validation signal is not
real training: trainable params > 0, >=20 steps, loss_final < loss_initial, and
plausible duration.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any


MIN_OPTIMIZER_STEPS = {MIN_OPTIMIZER_STEPS}
MIN_PLAUSIBLE_DURATION_S = {MIN_PLAUSIBLE_DURATION_S!r}
DEFAULT_BASE_MODEL = {base_model!r}
DEFAULT_CORPUS = Path({str(corpus_path)!r})
DEFAULT_OUTPUT = Path({str(output_path)!r})
DEFAULT_ADAPTER_DIR = Path({str(adapter_dir)!r})
DEFAULT_SEED = {int(random_seed)}
LORA_TARGET_MODULES = {target_modules!r}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty corpus: {{path}}")
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trainable_param_count(model: Any) -> int:
    return int(sum(int(param.numel()) for param in model.parameters() if getattr(param, "requires_grad", False)))


def validate_training_artifact(result: dict[str, Any], *, min_steps: int, min_duration_s: float) -> None:
    errors: list[str] = []
    if "qwen" in str(result.get("base_model", "")).lower():
        errors.append("trained_base_must_be_non_qwen")
    if result.get("model_load_api") != "AutoModelForCausalLM":
        errors.append("model_must_load_through_AutoModelForCausalLM")
    if not result.get("lora_attached"):
        errors.append("lora_not_attached")
    if int(result.get("trainable_param_count") or 0) <= 0:
        errors.append("no_trainable_lora_params")
    if int(result.get("optimizer_steps") or 0) < int(min_steps):
        errors.append("insufficient_optimizer_steps")
    loss_initial = result.get("loss_initial")
    loss_final = result.get("loss_final")
    if loss_initial is None or loss_final is None:
        errors.append("missing_loss_trace")
    elif not float(loss_final) < float(loss_initial):
        errors.append("loss_final < loss_initial validation failed")
    if float(result.get("duration_s") or 0.0) < float(min_duration_s):
        errors.append("duration_below_plausibility_floor")
    if errors:
        raise RuntimeError("real_training_validation_failed: " + ", ".join(errors))


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if "qwen" in args.base_model.lower():
        raise RuntimeError("Qwen is forbidden as the trained base for Exp 4263")

    started = time.time()
    rows = [row for row in load_rows(args.corpus) if float(row.get("reward_weight", 0.0)) > 0.0]
    if not rows:
        raise RuntimeError("no positive-weight training rows")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else None
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs: dict[str, Any] = {{"trust_remote_code": args.trust_remote_code}}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs).to(device)
    if hasattr(model, "config"):
        model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGET_MODULES,
        ),
    )
    trainable = trainable_param_count(model)
    if trainable <= 0:
        raise RuntimeError("LoRA attached with zero trainable parameters")
    model.train()
    optimizer = torch.optim.AdamW([param for param in model.parameters() if getattr(param, "requires_grad", False)], lr=args.learning_rate)
    loss_trace: list[dict[str, Any]] = []

    for step in range(args.min_steps):
        row = rows[step % len(rows)]
        prompt = str(row.get("prompt") or "")
        completion = str(row.get("completion") or "")
        text = prompt + "\\n" + completion + (tokenizer.eos_token or "")
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        labels = enc["input_ids"].clone()
        raw_loss = model(**enc, labels=labels).loss
        reward_weight = float(row.get("reward_weight", 1.0))
        loss = raw_loss * reward_weight
        if not bool(torch.isfinite(loss.detach()).all()):
            raise RuntimeError("non_finite_loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if getattr(param, "requires_grad", False)], 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_trace.append({{"step": step + 1, "loss": float(loss.detach().cpu()), "reward_weight": reward_weight, "arm_id": row.get("arm_id")}})

    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.adapter_dir)
    loss_initial = loss_trace[0]["loss"] if loss_trace else None
    loss_final = loss_trace[-1]["loss"] if loss_trace else None
    result = {{
        "base_model": args.base_model,
        "model_load_api": "AutoModelForCausalLM",
        "lora_attached": True,
        "trainable_param_count": trainable,
        "optimizer_steps": len(loss_trace),
        "loss_initial": loss_initial,
        "loss_final": loss_final,
        "loss_trace": loss_trace,
        "duration_s": round(time.time() - started, 6),
        "corpus_path": str(args.corpus),
        "adapter_dir": str(args.adapter_dir),
        "random_seed": args.seed,
    }}
    validate_training_artifact(result, min_steps=args.min_steps, min_duration_s=args.min_duration_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-steps", type=int, default=MIN_OPTIMIZER_STEPS)
    parser.add_argument("--min-duration-s", type=float, default=MIN_PLAUSIBLE_DURATION_S)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    result = run_training(args)
    print(json.dumps({{"validation_passed": True, "output": str(args.output), "loss_initial": result["loss_initial"], "loss_final": result["loss_final"]}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def write_out_of_band_runner(
    *,
    package_dir: str | Path,
    corpus_path: str | Path,
    base_model: str = TRAINABLE_BASE_MODEL,
    random_seed: int = RANDOM_SEED,
) -> tuple[Path, str, Path]:
    package = Path(package_dir)
    runner_path = package / RUNNER_FILENAME
    output_path = package / TRAINING_RESULT_FILENAME
    adapter_dir = package / "lora_adapter"
    script = _runner_script_text(
        corpus_path=Path(corpus_path),
        output_path=output_path,
        adapter_dir=adapter_dir,
        base_model=base_model,
        random_seed=random_seed,
    )
    package.mkdir(parents=True, exist_ok=True)
    runner_path.write_text(script, encoding="utf-8")
    runner_path.chmod(runner_path.stat().st_mode | stat.S_IXUSR)
    invocation = (
        f"python3 {runner_path} --corpus {Path(corpus_path)} --output {output_path} "
        f"--adapter-dir {adapter_dir} --base-model {base_model} --min-steps {MIN_OPTIMIZER_STEPS} "
        f"--min-duration-s {MIN_PLAUSIBLE_DURATION_S}"
    )
    return runner_path, invocation, output_path


def _model_specs(
    *,
    corpus_path: str | Path | None,
    runner_path: str | Path | None,
    training_result_path: str | Path | None,
    checksum: str,
) -> dict[str, Any]:
    return {
        "trainable_base": TRAINABLE_BASE_MODEL,
        "trainable_base_is_non_qwen": "qwen" not in TRAINABLE_BASE_MODEL.lower(),
        "trainable_base_is_gguf": "gguf" in TRAINABLE_BASE_MODEL.lower(),
        "qwen_train_base_forbidden": True,
        "load_method": f'transformers.AutoModelForCausalLM.from_pretrained("{TRAINABLE_BASE_MODEL}")',
        "reward_weighted_corpus_path": str(corpus_path or ""),
        "reward_weighted_corpus_checksum": checksum,
        "out_of_band_runner_path": str(runner_path or ""),
        "training_result_path": str(training_result_path or ""),
        "training_result_written_by_exp4263": False,
        "runner_config": {
            "min_optimizer_steps": MIN_OPTIMIZER_STEPS,
            "min_plausible_duration_s": MIN_PLAUSIBLE_DURATION_S,
            "validation_requires": [
                "AutoModelForCausalLM load",
                "LoRA attached",
                "trainable_param_count > 0",
                "optimizer_steps >= 20",
                "loss_final < loss_initial",
                "duration_s >= plausibility floor",
            ],
        },
        "lora_config": {
            "method": "LoRA-SFT",
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "learning_rate": 0.0002,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "task_type": "CAUSAL_LM",
        },
        "offline_reward_weighting_scheme": reward_weighting_scheme(),
    }


def _acceptance_satisfied(artifact: Mapping[str, Any]) -> bool:
    if artifact.get("honest_verdict") == "blocked_abc_corpora_missing":
        return True
    if bool(artifact.get("verifier_as_reward_retired")):
        return bool(artifact.get("verifier_is_oracle"))
    return bool(
        artifact.get("ready_for_out_of_band")
        and artifact.get("out_of_band_runner_path")
        and artifact.get("verifier_is_oracle") is True
    )


def write_manifest(
    *,
    package_dir: str | Path,
    corpus_path: str | Path,
    runner_path: str | Path,
    invocation: str,
    checksum: str,
    weighted: WeightedCorpus,
) -> Path:
    manifest_path = Path(package_dir) / PACKAGE_MANIFEST_FILENAME
    manifest = {
        "schema": "carnot.experiment_4263.out_of_band_package_manifest.v1",
        "corpus_path": str(corpus_path),
        "runner_path": str(runner_path),
        "one_command_invocation": invocation,
        "expected_runtime": "Approximately 5-20 minutes on a single CUDA GPU for the small non-Qwen smoke; CPU or first-time model download can take longer.",
        "reproducibility_checksum": checksum,
        "corpus_sizes": weighted.corpus_sizes,
        "reward_weight_counts": weighted.reward_weight_counts,
        "validation_gate": "LoRA trainable params >0, optimizer_steps >=20, loss_final < loss_initial, duration plausibility floor.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def build_artifact(
    *,
    verdict: str,
    bundle: CorpusBundle,
    weighted: WeightedCorpus,
    ready_for_out_of_band: bool,
    verifier_as_reward_retired: bool,
    corpus_path: str | Path | None,
    runner_path: str | Path | None,
    training_result_path: str | Path | None,
    invocation: str,
    checksum: str,
    package_manifest_path: str | Path | None,
    duration_s: float,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "experiment": "experiment_4263_verifier_as_reward_out_of_band_or_retire",
        "schema": "carnot.experiment_4263_verifier_as_reward_out_of_band_or_retire.v1",
        "honest_verdict": verdict,
        "ready_for_out_of_band": bool(ready_for_out_of_band),
        "verifier_as_reward_retired": bool(verifier_as_reward_retired),
        "out_of_band_runner_path": str(runner_path or ""),
        "verifier_is_oracle": True,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": checksum,
        "model_specs": _model_specs(
            corpus_path=corpus_path,
            runner_path=runner_path,
            training_result_path=training_result_path,
            checksum=checksum,
        ),
        "one_command_invocation": invocation,
        "expected_runtime": (
            "Approximately 5-20 minutes on a single CUDA GPU for the small non-Qwen smoke; CPU or "
            "first-time model download can take longer."
        ),
        "package_manifest_path": str(package_manifest_path or ""),
        "weighted_corpus": {
            "path": str(corpus_path or ""),
            "rows": len(weighted.rows),
            "corpus_sizes": weighted.corpus_sizes,
            "reward_weight_counts": weighted.reward_weight_counts,
            "supports_clean_avsb": weighted.supports_clean_avsb,
            "avsb_diagnostic": weighted.avsb_diagnostic,
        },
        "preconditions": {
            "abc_corpora_loaded": bundle.ready,
            "abc_corpus_paths": bundle.corpus_paths,
            "abc_corpus_sizes": bundle.corpus_sizes,
            "missing_abc_corpora": bundle.missing,
            "qwen_train_base_forbidden": True,
            "trained_base_is_non_qwen": "qwen" not in TRAINABLE_BASE_MODEL.lower(),
            "trm_runs_touched": False,
            "training_run_by_exp4263": False,
        },
        "retirement_evidence": (
            {
                "experiment_id": "exp2837",
                "fover_memory_ablation_delta": 0.0185,
                "standing": "FoVer +0.0185 memory-ablation stands as the self-learning evidence.",
            }
            if verifier_as_reward_retired
            else {}
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "created_at": _utc_now(),
        "duration_s": round(float(duration_s), 6),
        "acceptance_gate": {
            "condition": (
                "ready_for_out_of_band reported with out_of_band_runner_path (and "
                "verifier_as_reward_retired if prep infeasible) AND verifier_is_oracle=true, OR an "
                "honest blocked_abc_corpora_missing verdict"
            ),
            "principle": (
                "After 7 in-window failures, the disciplined resolution is to move training "
                "out-of-band OR retire -- not an 8th doomed in-window rerun; either outcome is "
                "decision-grade and closes the owed axis."
            ),
            "satisfied": False,
        },
    }
    artifact["acceptance_gate"]["satisfied"] = _acceptance_satisfied(artifact)
    return artifact


def write_artifact(artifact: Mapping[str, Any], path: str | Path = DEFAULT_OUTPUT) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def run(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    stable_checkpoint_path: str | Path = DEFAULT_STABLE_CHECKPOINT,
    package_dir: str | Path = DEFAULT_PACKAGE_DIR,
) -> dict[str, Any]:
    started = time.time()
    bundle = load_abc_corpora(stable_checkpoint_path)
    weighted = build_reward_weighted_corpus(bundle, random_seed=RANDOM_SEED)
    verdict: str
    ready = False
    retired = False
    corpus_path: Path | None = None
    runner_path: Path | None = None
    training_result_path: Path | None = None
    manifest_path: Path | None = None
    invocation = ""
    checksum = checksum_rows(weighted.rows) if weighted.rows else ""

    if not bundle.ready:
        verdict = "blocked_abc_corpora_missing"
    elif not weighted.supports_clean_avsb:
        verdict = "complete: verifier_as_reward_retired_fover_memory_ablation_stands"
        retired = True
    else:
        package = Path(package_dir)
        corpus_path = package / WEIGHTED_CORPUS_FILENAME
        checksum = write_jsonl_with_checksum(weighted.rows, corpus_path)
        runner_path, invocation, training_result_path = write_out_of_band_runner(
            package_dir=package,
            corpus_path=corpus_path,
            base_model=TRAINABLE_BASE_MODEL,
            random_seed=RANDOM_SEED,
        )
        manifest_path = write_manifest(
            package_dir=package,
            corpus_path=corpus_path,
            runner_path=runner_path,
            invocation=invocation,
            checksum=checksum,
            weighted=weighted,
        )
        ready = bool(corpus_path.is_file() and runner_path.is_file() and invocation)
        verdict = "complete: ready_for_out_of_band_verifier_reward_training"

    artifact = build_artifact(
        verdict=verdict,
        bundle=bundle,
        weighted=weighted,
        ready_for_out_of_band=ready,
        verifier_as_reward_retired=retired,
        corpus_path=corpus_path,
        runner_path=runner_path,
        training_result_path=training_result_path,
        invocation=invocation,
        checksum=checksum,
        package_manifest_path=manifest_path,
        duration_s=time.time() - started,
    )
    write_artifact(artifact, output_path)
    return _jsonable(artifact)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stable-checkpoint", type=Path, default=DEFAULT_STABLE_CHECKPOINT)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    args = parser.parse_args(argv)
    artifact = run(
        output_path=args.out,
        stable_checkpoint_path=args.stable_checkpoint,
        package_dir=args.package_dir,
    )
    print(f"-> {artifact['honest_verdict']}", flush=True)
    print(f"   ready_for_out_of_band={artifact['ready_for_out_of_band']}", flush=True)
    print(f"   verifier_as_reward_retired={artifact['verifier_as_reward_retired']}", flush=True)
    print(f"   out_of_band_runner_path={artifact['out_of_band_runner_path']}", flush=True)
    return 0 if artifact["acceptance_gate"]["satisfied"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main(sys.argv[1:]))
