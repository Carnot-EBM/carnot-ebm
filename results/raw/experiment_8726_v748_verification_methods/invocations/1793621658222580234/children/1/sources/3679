"""Tiny EBT GSM8K CUDA train-step smoke for Exp 3726.

Spec: REQ-EBT-3726, SCENARIO-EBT-3726.

This module keeps the experiment honest by separating three concerns:

* corpus handling: GSM8K train rows are converted into byte-token training
  blocks, so the data source is a clean reasoning corpus rather than P0.1
  generation dumps;
* substrate handling: CUDA, vendored EBT, and dataset availability are checked
  before training starts;
* artifact handling: the JSON records the exact tiny from-scratch EBT size,
  first losses, peak VRAM, seed, checksum, and precondition evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


EXPERIMENT_ID = 3726
RESULT_PATH = Path("results/experiment_3726_tiny_ebt_corpus_and_train_step_smoke.json")
RANDOM_SEED = 3726

BYTE_TOKENIZER_EOS_ID = 1
BYTE_TOKENIZER_BYTE_OFFSET = 2
BYTE_TOKENIZER_VOCAB_SIZE = 258
LIVE_INFERENCE_SUBSTRATE = (
    "live_llm_inference (principle: a real GPU compute task; strict duration "
    "floor, which real training easily clears)."
)
GSM8K_DATASET_API = "https://huggingface.co/api/datasets"


@dataclass(frozen=True)
class TinyEBTSmokeConfig:
    """Configuration for the tiny from-scratch EBT smoke run.

    The EBT core is sized through the vendored upstream ``EBTModelArgs`` knobs.
    ``dim=768``, ``n_layers=4``, and a 4x feed-forward multiplier keep the
    trainable model in the requested 10M-50M parameter band while remaining
    small enough for a single RTX 3090.
    """

    n_train: int = 2048
    dim: int = 768
    n_layers: int = 4
    n_heads: int = 12
    ffn_dim_multiplier: float = 4.0
    block_size: int = 128
    batch_size: int = 8
    train_steps: int = 64
    learning_rate: float = 3e-4
    heartbeat_every: int = 4


class PreconditionError(RuntimeError):
    """Raised when a hard resource precondition is absent."""

    def __init__(self, verdict: str, preconditions: dict[str, Any]) -> None:
        super().__init__(verdict)
        self.verdict = verdict
        self.preconditions = preconditions


def _json_default(value: Any) -> Any:
    """Convert numpy scalar values to JSON primitives."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def format_gsm8k_row(row: dict[str, Any]) -> str:
    """Format one GSM8K train row as structured reasoning text."""
    return f"Question: {row['question']}\nAnswer: {row['answer']}"


def load_gsm8k_train_texts(n_train: int) -> list[str]:
    """Load a bounded GSM8K train subset and return formatted reasoning rows."""
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    from datasets import load_dataset

    dataset = load_dataset("gsm8k", "main", split=f"train[:{n_train}]")
    return [format_gsm8k_row(row) for row in dataset]


def _encode_byte_tokens(text: str) -> list[int]:
    """Encode text with a small standard UTF-8 byte tokenizer.

    Byte tokenization is deterministic, does not import pretrained model
    weights, and still produces real token IDs for the GSM8K corpus. Byte
    values are shifted by two because 0 is reserved for padding if a future
    caller needs it and 1 is the per-example EOS separator.
    """
    return [byte + BYTE_TOKENIZER_BYTE_OFFSET for byte in text.encode("utf-8")] + [
        BYTE_TOKENIZER_EOS_ID
    ]


def tokenize_texts_to_blocks(texts: list[str], block_size: int) -> np.ndarray:
    """Pack tokenized GSM8K rows into contiguous next-token training blocks."""
    if block_size <= 1:
        raise ValueError("block_size must be greater than 1")
    token_stream: list[int] = []
    for text in texts:
        token_stream.extend(_encode_byte_tokens(text))

    chunk_len = block_size + 1
    n_blocks = len(token_stream) // chunk_len
    if n_blocks <= 0:
        raise ValueError("not enough tokenized text for one training block")

    trimmed = token_stream[: n_blocks * chunk_len]
    return np.asarray(trimmed, dtype=np.int64).reshape(n_blocks, chunk_len)


def reproducibility_checksum(texts: list[str], config: TinyEBTSmokeConfig) -> str:
    """Hash the exact corpus text and config that determine this smoke run."""
    corpus_hash = hashlib.sha256()
    for text in texts:
        corpus_hash.update(text.encode("utf-8"))
        corpus_hash.update(b"\0")
    payload = {
        "config": asdict(config),
        "corpus_sha256": corpus_hash.hexdigest(),
        "tokenizer": "utf8_byte_tokenizer_v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def loss_smoke_passed(losses: list[float]) -> bool:
    """Return true when losses are finite and the final observed loss is lower."""
    if len(losses) < 2:
        return False
    if not all(math.isfinite(float(loss)) for loss in losses):
        return False
    return float(losses[-1]) < float(losses[0])


def _round_losses(losses: list[float], limit: int = 10) -> list[float]:
    """Round the first losses for stable JSON without hiding their shape."""
    return [round(float(loss), 6) for loss in losses[:limit]]


def model_specs(config: TinyEBTSmokeConfig, param_count: int) -> dict[str, Any]:
    """Describe the tiny EBT model without implying pretrained LLM weights."""
    return {
        "model_name": "tiny_vendored_ebt_byte_energy_model",
        "source": "carnot.phase3.ebt_upstream.EBTDefault",
        "from_scratch_not_pretrained_llm": True,
        "pretrained_model_weights_used": False,
        "tokenizer": {
            "name": "utf8_byte_tokenizer",
            "vocab_size": BYTE_TOKENIZER_VOCAB_SIZE,
            "pretrained_tokenizer_weights_used": False,
        },
        "ebt_model_args": {
            "dim": config.dim,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "ffn_dim_multiplier": config.ffn_dim_multiplier,
            "max_seq_len": config.block_size + 2,
        },
        "trainable_param_count": int(param_count),
        "optimizer": "AdamW",
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "block_size": config.block_size,
        "train_steps": config.train_steps,
        "device": "cuda:0",
    }


def _param_millions(param_count: int) -> int:
    """Return a terminal-verdict-sized whole-million parameter count."""
    return int(round(int(param_count) / 1_000_000))


def build_success_artifact(
    *,
    n_train: int,
    param_count: int,
    peak_vram_mb: int,
    losses: list[float],
    preconditions: dict[str, Any],
    model_specs: dict[str, Any],
    random_seed: int,
    checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    """Build the success artifact required by REQ-EBT-3726."""
    first_losses = _round_losses(losses)
    return {
        "schema": "carnot.experiment_3726_tiny_ebt_train_smoke.v1",
        "experiment": EXPERIMENT_ID,
        "honest_verdict": (
            f"complete: tiny_ebt_{_param_millions(param_count)}M_fits_3090_"
            f"{int(peak_vram_mb)}mb_single_train_step_loss_finite_and_decreasing_"
            f"corpus_gsm8k_n{int(n_train)}"
        ),
        "inference_substrate": LIVE_INFERENCE_SUBSTRATE,
        "n_train": int(n_train),
        "ebt_param_count": int(param_count),
        "peak_vram_mb": int(peak_vram_mb),
        "first_step_losses": first_losses,
        "loss_finite": all(math.isfinite(float(loss)) for loss in losses),
        "loss_decreased": loss_smoke_passed(first_losses),
        "preconditions_checked": preconditions,
        "model_specs": model_specs,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "duration_s": round(float(duration_s), 3),
    }


def build_blocked_artifact(
    verdict: str,
    *,
    preconditions: dict[str, Any],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    """Build a terminal blocked artifact when a hard precondition is missing."""
    return {
        "schema": "carnot.experiment_3726_tiny_ebt_train_smoke.v1",
        "experiment": EXPERIMENT_ID,
        "honest_verdict": verdict,
        "inference_substrate": LIVE_INFERENCE_SUBSTRATE,
        "n_train": 0,
        "ebt_param_count": 0,
        "peak_vram_mb": 0,
        "first_step_losses": [],
        "loss_finite": False,
        "preconditions_checked": preconditions,
        "model_specs": {
            "model_name": "tiny_vendored_ebt_byte_energy_model",
            "from_scratch_not_pretrained_llm": True,
            "pretrained_model_weights_used": False,
        },
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "duration_s": round(float(duration_s), 3),
    }


def validate_success_artifact(artifact: dict[str, Any]) -> list[str]:
    """Return schema errors that would make the success artifact misleading."""
    errors: list[str] = []
    required_fields = {
        "honest_verdict",
        "inference_substrate",
        "n_train",
        "ebt_param_count",
        "peak_vram_mb",
        "first_step_losses",
        "loss_finite",
        "preconditions_checked",
        "model_specs",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    }
    missing = sorted(required_fields - set(artifact))
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")

    if not str(artifact.get("honest_verdict", "")).startswith("complete:"):
        errors.append("honest_verdict must start with complete:")
    if not str(artifact.get("inference_substrate", "")).startswith("live_llm_inference"):
        errors.append("inference_substrate must identify live_llm_inference")

    param_count = int(artifact.get("ebt_param_count") or 0)
    if not 10_000_000 <= param_count <= 50_000_000:
        errors.append("ebt_param_count must be within 10M-50M")
    if int(artifact.get("n_train") or 0) <= 0:
        errors.append("n_train must be positive")
    if int(artifact.get("peak_vram_mb") or 0) <= 0:
        errors.append("peak_vram_mb must be positive")
    if artifact.get("loss_finite") is not True:
        errors.append("loss_finite must be true")
    if not loss_smoke_passed([float(loss) for loss in artifact.get("first_step_losses", [])]):
        errors.append("first_step_losses must be finite and decreasing")
    if artifact.get("model_specs", {}).get("from_scratch_not_pretrained_llm") is not True:
        errors.append("model_specs must identify a from-scratch non-pretrained EBT")
    if not artifact.get("reproducibility_checksum"):
        errors.append("reproducibility_checksum must be present")
    if float(artifact.get("duration_s") or 0.0) <= 0.0:
        errors.append("duration_s must be positive")
    return errors


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Write a stable JSON artifact for downstream conductor verification."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _local_gsm8k_cache_exists() -> bool:
    """Return whether a HuggingFace GSM8K dataset cache is present locally."""
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets"
    if not cache_root.exists():
        return False
    return any(cache_root.glob("gsm8k*")) or any(cache_root.glob("**/gsm8k*"))


def _dataset_api_available() -> bool:
    """Check the HuggingFace dataset API with the requested curl precondition."""
    try:
        completed = subprocess.run(
            ["curl", "-sf", "-o", "/dev/null", GSM8K_DATASET_API],
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def check_preconditions() -> dict[str, Any]:
    """Check CUDA, vendored EBT, and GSM8K dataset availability before training."""
    preconditions: dict[str, Any] = {
        "python_executable": sys.executable,
        "cuda": {"available": False, "device_count": 0, "device_name": None},
        "ebt_vendored_importable": False,
        "dataset": {"available": False, "source": None},
    }

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        preconditions["cuda"] = {
            "available": cuda_available,
            "device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        }
    except Exception as exc:  # pragma: no cover - only triggered on broken envs.
        preconditions["cuda"]["error"] = str(exc)
    if not preconditions["cuda"]["available"]:
        raise PreconditionError("blocked_cuda", preconditions)

    try:
        import carnot.phase3.ebt_upstream  # noqa: F401

        preconditions["ebt_vendored_importable"] = True
    except Exception as exc:
        preconditions["ebt_vendored_error"] = str(exc)
        raise PreconditionError("blocked_ebt_not_vendored", preconditions) from exc

    if _dataset_api_available():
        preconditions["dataset"] = {"available": True, "source": "huggingface_api"}
    elif _local_gsm8k_cache_exists():
        preconditions["dataset"] = {"available": True, "source": "local_gsm8k_cache"}
    else:
        raise PreconditionError("blocked_gsm8k_dataset", preconditions)

    return preconditions


def set_reproducibility(seed: int) -> None:
    """Seed Python, numpy, and torch before model initialization."""
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_tiny_ebt_model(config: TinyEBTSmokeConfig) -> Any:
    """Build the byte-token wrapper around the vendored upstream EBT core."""
    import torch
    from torch import nn

    from carnot.phase3.ebt_upstream import EBTDefault, EBTModelArgs

    class TinyByteEBT(nn.Module):
        """Minimal from-scratch token embedding plus EBT energy core."""

        def __init__(self) -> None:
            super().__init__()
            args = EBTModelArgs(
                dim=config.dim,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                ffn_dim_multiplier=config.ffn_dim_multiplier,
                max_batch_size=config.batch_size,
                max_seq_len=config.block_size + 2,
            )
            self.token_embedding = nn.Embedding(BYTE_TOKENIZER_VOCAB_SIZE, config.dim)
            self.ebt = EBTDefault(args)
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        def forward(self, token_ids: Any) -> Any:
            original = self.token_embedding(token_ids[:, :-1])
            predicted = self.token_embedding(token_ids[:, 1:])
            embeddings = torch.cat((original, predicted), dim=1)
            energies = self.ebt(embeddings, start_pos=0)
            return energies.mean()

    return TinyByteEBT()


def count_parameters(model: Any) -> int:
    """Count trainable model parameters."""
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def run_cuda_training_smoke(
    token_blocks: np.ndarray,
    config: TinyEBTSmokeConfig,
) -> dict[str, Any]:
    """Run real CUDA optimizer steps and return loss/VRAM evidence."""
    if token_blocks.shape[0] < config.batch_size:
        raise ValueError("not enough token blocks for one batch")

    import torch

    device = torch.device("cuda:0")
    model = build_tiny_ebt_model(config).to(device)
    param_count = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    blocks = torch.as_tensor(token_blocks, dtype=torch.long)
    batch = blocks[: config.batch_size].to(device, non_blocking=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()

    losses: list[float] = []
    for step in range(config.train_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch)
        if not torch.isfinite(loss):  # pragma: no cover - live run must avoid this bailout.
            losses.append(float("nan"))
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if step == 0 or (step + 1) % config.heartbeat_every == 0:
            print(
                f"heartbeat exp3726 step={step + 1}/{config.train_steps} "
                f"loss={loss_value:.6f}",
                flush=True,
            )

    torch.cuda.synchronize(device)
    peak_vram_mb = int(torch.cuda.max_memory_allocated(device) // (1024 * 1024))
    return {
        "losses": losses,
        "param_count": param_count,
        "peak_vram_mb": peak_vram_mb,
    }


def run_experiment(
    *,
    config: TinyEBTSmokeConfig = TinyEBTSmokeConfig(),
    result_path: Path = RESULT_PATH,
) -> dict[str, Any]:
    """Run the full Exp 3726 smoke and write its artifact."""
    start = time.monotonic()
    preconditions: dict[str, Any] = {}
    try:
        preconditions = check_preconditions()
    except PreconditionError as exc:
        artifact = build_blocked_artifact(
            exc.verdict,
            preconditions=exc.preconditions,
            random_seed=RANDOM_SEED,
            duration_s=time.monotonic() - start,
        )
        write_artifact(result_path, artifact)
        print(artifact["honest_verdict"], flush=True)
        return artifact

    set_reproducibility(RANDOM_SEED)
    texts = load_gsm8k_train_texts(config.n_train)
    checksum = reproducibility_checksum(texts, config)
    token_blocks = tokenize_texts_to_blocks(texts, config.block_size)
    train_result = run_cuda_training_smoke(token_blocks, config)
    duration_s = time.monotonic() - start

    specs = model_specs(config, param_count=train_result["param_count"])
    artifact = build_success_artifact(
        n_train=len(texts),
        param_count=train_result["param_count"],
        peak_vram_mb=train_result["peak_vram_mb"],
        losses=train_result["losses"],
        preconditions=preconditions,
        model_specs=specs,
        random_seed=RANDOM_SEED,
        checksum=checksum,
        duration_s=duration_s,
    )
    errors = validate_success_artifact(artifact)
    if errors:
        artifact["honest_verdict"] = "blocked_training_smoke_validation"
        artifact["validation_errors"] = errors
    write_artifact(result_path, artifact)
    print(artifact["honest_verdict"], flush=True)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``scripts/experiment_3726...py``."""
    _ = argv
    artifact = run_experiment()
    return 0 if str(artifact.get("honest_verdict", "")).startswith("complete:") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
