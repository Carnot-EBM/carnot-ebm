"""Reusable LoRA-EBM "moat trainer" — the shared library the headline moat
experiment (D1) imports instead of re-deriving a training pipeline each time.

Why this module exists
----------------------
The D1 headline (a trained energy verifier that beats self-consistency on a
reasoning corpus) bailed TWICE before this module shipped:

  * `.461` bailed skeleton-first (it wrote a 0-pair skeleton and never trained).
  * `.462` bailed on a 404: the planner named ``Qwen/Qwen3.5-1.7B`` as the
    trainable base, but that repo does NOT exist on HuggingFace.  A single
    hallucinated repo id blocked the whole headline.

This module fixes the *class* of bug, not just the one instance.  Three pieces:

  1. ``resolve_trainable_base`` — never trust a single (possibly hallucinated)
     repo id again.  Probe a PRIORITIZED list of bases that are CONFIRMED
     cached on the conductor box and return the first one that actually has
     weights on disk.  A wrong id can no longer block training.

  2. ``train_energy_head`` — load that real base, attach a LoRA adapter plus a
     scalar "energy head", and train it CONTRASTIVELY so good reasoning gets a
     LOWER energy than bad reasoning.  Checkpoint per epoch so a capped /
     interrupted run RESUMES from the last epoch instead of starting over.

  3. ``score_candidates`` — load a trained checkpoint and return one scalar
     energy per candidate (lower = better).  This is the eval path D1 uses to
     rank candidate answers.

The training recipe (QLoRA 4-bit base + a ``num_labels=1`` sequence-regression
head as the scalar energy, contrastive ``softplus(E_good - E_bad)`` loss,
per-epoch adapter checkpoints) is the SAME recipe that already trains cleanly
in ``experiment_5017_lora_ebm_scorer_musr_v2`` — we are extracting it into a
reusable, import-once module, not inventing a new one.  The energy convention
is: ``energy(good) < energy(bad)`` (low energy = high quality), so a selector
picks ``argmin`` energy.

Spec: REQ-VERIFY-5030, SCENARIO-VERIFY-5030.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

# The prioritized list of trainable bases CONFIRMED cacheable on the conductor
# box.  ``resolve_trainable_base`` walks this in order and returns the first one
# whose local HuggingFace snapshot actually has weights.  Every id here is a
# REAL HuggingFace repo (unlike the hallucinated ``Qwen/Qwen3.5-1.7B`` that 404'd
# the `.462` headline).  Smallest-useful first would be faster, but the headline
# wants the strongest cached base it can get, so the 2B leads and the 0.5B is the
# always-present fallback.
PRIORITY_BASES: tuple[str, ...] = (
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen2.5-0.5B",
)

JsonDict = dict[str, Any]


# --------------------------------------------------------------------------- #
# Base resolution — the fix for the hallucinated-repo failure class.
# --------------------------------------------------------------------------- #
def default_hub_cache() -> Path:
    """Return the HuggingFace hub cache directory on this machine.

    Honours ``HF_HOME`` / ``HF_HUB_CACHE`` if the operator set them, else falls
    back to the standard ``~/.cache/huggingface/hub``.  Kept as its own function
    so tests can probe the real cache without monkeypatching environment.
    """
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_cache_dir_name(repo_id: str) -> str:
    """Map a repo id (``org/name``) to its on-disk cache folder name.

    HuggingFace stores ``Qwen/Qwen3.5-2B`` under ``models--Qwen--Qwen3.5-2B``.
    """
    return "models--" + repo_id.replace("/", "--")


def snapshot_with_weights(repo_id: str, *, hub_cache: Path | None = None) -> Path | None:
    """Return the local snapshot dir for ``repo_id`` IF it has usable weights.

    "Usable" means the snapshot has a ``config.json`` AND at least one
    safetensors weight file (a bare ``*.safetensors`` shard or the
    ``model.safetensors.index.json`` manifest for a sharded model).  Returns
    ``None`` when the model is not cached or the cache is incomplete (e.g. only
    metadata was fetched).  This is the per-base presence check
    ``resolve_trainable_base`` walks.
    """
    cache_root = hub_cache or default_hub_cache()
    model_dir = Path(cache_root) / _hf_cache_dir_name(repo_id)
    snap_root = model_dir / "snapshots"
    if not snap_root.is_dir():
        return None
    # Newest snapshot first so we prefer the most-recently-fetched revision.
    for snapshot in sorted(snap_root.glob("*"), reverse=True):
        if not snapshot.is_dir():
            continue
        has_config = (snapshot / "config.json").exists()
        has_weights = (
            bool(list(snapshot.glob("*.safetensors")))
            or (snapshot / "model.safetensors.index.json").exists()
        )
        if has_config and has_weights:
            return snapshot
    return None


def resolve_trainable_base(
    preferred: str | None = None,
    *,
    hub_cache: Path | None = None,
    priority: Sequence[str] = PRIORITY_BASES,
) -> tuple[str, str]:
    """Return ``(repo_id, local_snapshot_path)`` for the first REAL cached base.

    This is the load-bearing fix.  Instead of trusting a single (possibly
    hallucinated) repo id, we probe a prioritized list of bases that are known
    to be cacheable here and return the FIRST whose local snapshot actually has
    weights on disk.

    Parameters
    ----------
    preferred:
        An optional caller-preferred repo id (e.g. what a roadmap task named).
        If it is present on disk it is used first; otherwise we silently fall
        through to the priority list — so a wrong/hallucinated ``preferred`` can
        NEVER block training, it just gets skipped.
    hub_cache:
        Override the hub cache dir (used by tests).  Defaults to the real cache.
    priority:
        The fallback probe order.  Defaults to ``PRIORITY_BASES``.

    Raises
    ------
    RuntimeError
        Only when NONE of the probed bases is present — and the message names
        the full list that was probed, so the failure is actionable.
    """
    # Build the probe order: preferred first (deduped), then the priority list.
    probe_order: list[str] = []
    if preferred:
        probe_order.append(preferred)
    for repo_id in priority:
        if repo_id not in probe_order:
            probe_order.append(repo_id)

    for repo_id in probe_order:
        snapshot = snapshot_with_weights(repo_id, hub_cache=hub_cache)
        if snapshot is not None:
            return repo_id, snapshot.as_posix()

    raise RuntimeError(
        "no trainable base cached: probed "
        + ", ".join(probe_order)
        + f" under {Path(hub_cache or default_hub_cache()).as_posix()}; "
        "cache at least one (e.g. Qwen/Qwen3.5-2B) before training"
    )


# --------------------------------------------------------------------------- #
# Pair normalization — accept tuples or objects with .good_text/.bad_text.
# --------------------------------------------------------------------------- #
def normalize_pairs(pairs: Sequence[Any]) -> list[tuple[str, str]]:
    """Coerce ``pairs`` to a clean list of ``(good_text, bad_text)`` tuples.

    Accepts either 2-tuples ``(good, bad)`` or objects exposing ``.good_text`` /
    ``.bad_text`` (the ``TrainingPair`` shape used elsewhere in the repo), so a
    caller can hand us whatever pair representation it already has.  Empty / blank
    entries are dropped — a pair with no contrastive signal teaches nothing.
    """
    out: list[tuple[str, str]] = []
    for item in pairs:
        good = getattr(item, "good_text", None)
        bad = getattr(item, "bad_text", None)
        if good is None or bad is None:
            # Fall back to a 2-element sequence (tuple/list).
            seq = list(item)
            if len(seq) < 2:
                continue
            good, bad = seq[0], seq[1]
        good_s = str(good).strip()
        bad_s = str(bad).strip()
        if good_s and bad_s:
            out.append((good_s, bad_s))
    return out


# --------------------------------------------------------------------------- #
# Checkpoint helpers (pure — safe to unit test without a GPU).
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, payload: JsonDict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def latest_epoch_checkpoint(out_dir: Path) -> Path | None:
    """Return the highest-numbered ``epoch_N`` dir under ``out_dir`` with metrics.

    Used both to RESUME a capped run (continue from the last finished epoch) and
    to let ``score_candidates`` find the trained adapter when handed the parent
    directory.  Returns ``None`` when nothing has been checkpointed yet.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in out_dir.glob("epoch_*"):
        if not child.is_dir():
            continue
        if not (child / "train_metrics.json").exists():
            continue
        try:
            epoch_n = int(child.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates.append((epoch_n, child))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def reproducibility_checksum(
    base_used: str,
    pairs: Sequence[tuple[str, str]],
    *,
    lora_r: int,
    lora_alpha: int,
    seed: int,
) -> str:
    """Content hash of (base, LoRA config, pairs, seed).

    A replication that drifts on any of these inputs produces a different hash,
    so a downstream consumer can detect "this checkpoint was trained on different
    data/config than I expected" instead of silently comparing apples to oranges.
    """
    hasher = hashlib.sha256()
    hasher.update(base_used.encode("utf-8"))
    hasher.update(f"|r={lora_r}|a={lora_alpha}|seed={seed}|n={len(pairs)}|".encode())
    for good, bad in pairs:
        hasher.update(good.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(bad.encode("utf-8"))
        hasher.update(b"\x01")
    return "sha256:" + hasher.hexdigest()


def auto_load_in_4bit() -> bool:
    """Decide whether to use 4-bit QLoRA: yes iff bitsandbytes + CUDA are present.

    4-bit loading drops a 2B base from ~8 GB (fp32) to ~1.5 GB and loads faster,
    which is what makes the smoke and D1 fit comfortably on one 3090.  When
    bitsandbytes is unavailable we fall back to a plain bf16 load so the module
    still works, just hungrier.
    """
    try:  # pragma: no cover - trivial import probe, environment-dependent
        import bitsandbytes  # noqa: F401
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


# --------------------------------------------------------------------------- #
# Live GPU stack loader (pragma no-cover: exercised by the test/smoke, but the
# branch matrix is environment-dependent so we exclude it from coverage gates,
# matching the repo convention in experiment_5017).
# --------------------------------------------------------------------------- #
def _load_base_stack(
    repo_id: str,
    local_path: str,
    *,
    load_in_4bit: bool,
    device_index: int,
    adapter_path: Path | None = None,
    trainable: bool = True,
):  # pragma: no cover - live transformer load
    """Load tokenizer + a ``num_labels=1`` regression model = the scalar energy.

    ``AutoModelForSequenceClassification(num_labels=1, problem_type="regression")``
    gives us a single scalar logit per input — that logit IS the energy.  The
    freshly-initialized ``score`` head is what LoRA trains (via
    ``modules_to_save=["score"]``).  When ``adapter_path`` is given we reload a
    previously-trained adapter on top of the base (for resume or scoring).
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "num_labels": 1,
        "problem_type": "regression",
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = {"": device_index}

    model = AutoModelForSequenceClassification.from_pretrained(local_path, **load_kwargs)
    if not load_in_4bit:
        model = model.to(f"cuda:{device_index}")
    _set_pad_token_id(model, tokenizer.pad_token_id)
    model.config.use_cache = False

    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, Path(adapter_path).as_posix(), is_trainable=trainable
        )
        # A sequence-classification model needs pad_token_id on the DEEPEST config
        # (it locates the last non-pad token); the peft wrap can shadow it.
        _set_pad_token_id(model, tokenizer.pad_token_id)
    return tokenizer, model


def _set_pad_token_id(
    model: Any, pad_token_id: int | None
) -> None:  # pragma: no cover - live model config
    """Set ``pad_token_id`` on every config layer including the base model.

    The sequence-classification forward needs ``config.pad_token_id`` to find the
    last real token; after a peft wrap the PeftModel's ``config`` can differ from
    the inner transformer's, so we walk down via ``get_base_model`` too.
    """
    if pad_token_id is None:
        return
    targets = [model]
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        try:
            targets.append(getter())
        except Exception:  # noqa: BLE001
            pass
    targets.append(getattr(model, "base_model", None))
    for target in targets:
        cfg = getattr(target, "config", None)
        if cfg is None:
            continue
        cfg.pad_token_id = pad_token_id
        # Modern multimodal configs (e.g. Qwen3.5) keep the real pad_token_id on a
        # text sub-config; the seq-classification forward reads
        # ``config.get_text_config().pad_token_id``, so set it there too.
        text_getter = getattr(cfg, "get_text_config", None)
        if callable(text_getter):
            try:
                text_cfg = text_getter()
            except Exception:  # noqa: BLE001
                text_cfg = None
            if text_cfg is not None and text_cfg is not cfg:
                text_cfg.pad_token_id = pad_token_id


def train_energy_head(
    base: str | tuple[str, str],
    pairs: Sequence[Any],
    out_dir: str | Path,
    *,
    epochs: int = 1,
    lr: float = 2e-4,
    batch_size: int = 2,
    max_length: int = 256,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    max_steps: int | None = None,
    device_index: int = 0,
    load_in_4bit: bool | None = None,
    seed: int = 0,
) -> JsonDict:  # pragma: no cover - live GPU training path (exercised by tests)
    """Train a LoRA + scalar-energy-head verifier contrastively and checkpoint.

    The model learns ``energy(good) < energy(bad)`` via a ``softplus(E_good -
    E_bad)`` loss: when the good candidate already has lower energy the loss is
    near zero, otherwise it pushes the good energy down / bad energy up.  After
    each epoch the adapter + tokenizer + a ``train_metrics.json`` marker are
    saved to ``out_dir/epoch_N`` so a capped or interrupted run RESUMES from the
    last finished epoch instead of starting over.

    Parameters
    ----------
    base:
        Either a repo id (string — resolved via ``resolve_trainable_base`` so a
        wrong id falls through to a real cached base) or a pre-resolved
        ``(repo_id, local_path)`` tuple.
    pairs:
        Contrastive ``(good_text, bad_text)`` pairs (tuples or objects with
        ``.good_text`` / ``.bad_text``).
    out_dir:
        Checkpoint directory.
    epochs, lr, batch_size, max_length, lora_*, max_steps, device_index,
    load_in_4bit, seed:
        Standard training knobs.  ``max_steps`` caps total optimizer steps (for
        a bounded smoke).  ``device_index`` is the CUDA device — 0 = conductor
        GPU 0 (the 2026-06-27 GPU fix: train on GPU 0, never iGPU-pin).

    Returns
    -------
    dict with ``train_loss``, ``n_pairs``, ``base_used``, ``checkpoint_dir``,
    ``epochs_done``, ``n_steps``, ``loss_history``, ``resumed``,
    ``reproducibility_checksum`` and ``model_specs``.
    """
    import random as _random

    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if isinstance(base, tuple):
        repo_id, local_path = base
    else:
        repo_id, local_path = resolve_trainable_base(preferred=base)

    train_pairs = normalize_pairs(pairs)
    if not train_pairs:
        raise RuntimeError("no_contrastive_pairs: train_energy_head needs >=1 (good, bad) pair")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if load_in_4bit is None:
        load_in_4bit = auto_load_in_4bit()

    # ---- Resume bookkeeping: continue from the last finished epoch. ---------
    start_epoch = 0
    resume_adapter: Path | None = None
    prior_losses: list[float] = []
    latest = latest_epoch_checkpoint(out_dir)
    if latest is not None:
        prior = _read_json(latest / "train_metrics.json")
        if isinstance(prior, dict) and prior.get("train_loss") is not None:
            start_epoch = int(prior.get("epochs_done", 0))
            prior_losses = [float(x) for x in prior.get("loss_history", [])]
            if start_epoch >= epochs:
                # Already trained for >= the requested epochs: return as-is.
                return {
                    "train_loss": float(prior["train_loss"]),
                    "n_pairs": int(prior.get("n_pairs", len(train_pairs))),
                    "base_used": repo_id,
                    "checkpoint_dir": latest.as_posix(),
                    "epochs_done": start_epoch,
                    "n_steps": int(prior.get("n_steps", len(prior_losses))),
                    "loss_history": prior_losses,
                    "resumed": True,
                    "reproducibility_checksum": prior.get("reproducibility_checksum"),
                    "model_specs": prior.get("model_specs", {}),
                }
            resume_adapter = latest  # pragma: no cover - partial-resume continuation

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = _random.Random(seed)

    tokenizer, model = _load_base_stack(
        repo_id,
        local_path,
        load_in_4bit=load_in_4bit,
        device_index=device_index,
        adapter_path=resume_adapter,
        trainable=True,
    )
    if resume_adapter is None:
        if load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        lora = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=["score"],
        )
        model = get_peft_model(model, lora)
    _set_pad_token_id(model, tokenizer.pad_token_id)
    for wrapped in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        if wrapped is not None and getattr(wrapped, "config", None) is not None:
            wrapped.config.use_cache = False

    model.train()
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad), lr=lr
    )

    losses: list[float] = list(prior_losses)
    total_steps = len(prior_losses)
    checkpoint_dir = latest.as_posix() if latest is not None else out_dir.as_posix()
    specs = {
        "base_model": repo_id,
        "base_cache_path": local_path,
        "adapter": "LoRA",
        "energy_head": "scalar_sequence_regression_head",
        "quantization": "4bit_nf4_QLoRA" if load_in_4bit else "bf16",
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "max_length": max_length,
    }
    checksum = reproducibility_checksum(
        repo_id, train_pairs, lora_r=lora_r, lora_alpha=lora_alpha, seed=seed
    )

    stop = False
    for epoch in range(start_epoch, epochs):
        rng.shuffle(train_pairs)
        for batch_start in range(0, len(train_pairs), batch_size):
            if max_steps is not None and total_steps >= max_steps:
                stop = True
                break
            batch = train_pairs[batch_start : batch_start + batch_size]
            good_inputs = tokenizer(
                [g for g, _ in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            bad_inputs = tokenizer(
                [b for _, b in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            good_energy = model(**good_inputs).logits.float().view(-1)
            bad_energy = model(**bad_inputs).logits.float().view(-1)
            loss = torch.nn.functional.softplus(good_energy - bad_energy).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            total_steps += 1

        epoch_dir = out_dir / f"epoch_{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir.as_posix())
        tokenizer.save_pretrained(epoch_dir.as_posix())
        running_loss = sum(losses) / len(losses) if losses else math.nan
        _write_json(
            epoch_dir / "train_metrics.json",
            {
                "train_loss": round(running_loss, 6),
                "n_pairs": len(train_pairs),
                "epochs_done": epoch + 1,
                "n_steps": total_steps,
                "loss_history": [round(x, 6) for x in losses],
                "model_specs": specs,
                "reproducibility_checksum": checksum,
            },
        )
        checkpoint_dir = epoch_dir.as_posix()
        if stop:
            break

    model.eval()
    train_loss = sum(losses) / len(losses) if losses else math.nan
    epochs_done = epoch + 1 if epochs > start_epoch else start_epoch
    return {
        "train_loss": round(train_loss, 6) if losses else None,
        "n_pairs": len(train_pairs),
        "base_used": repo_id,
        "checkpoint_dir": checkpoint_dir,
        "epochs_done": epochs_done,
        "n_steps": total_steps,
        "loss_history": [round(x, 6) for x in losses],
        "resumed": resume_adapter is not None,
        "reproducibility_checksum": checksum,
        "model_specs": specs,
    }


def score_candidates(
    checkpoint: str | Path,
    candidate_texts: Sequence[str],
    *,
    max_length: int = 256,
    device_index: int = 0,
    load_in_4bit: bool | None = None,
) -> list[float]:  # pragma: no cover - live GPU scoring path (exercised by tests)
    """Load a trained checkpoint and return one scalar energy per candidate.

    Lower energy = higher quality (the contrastive loss trained good reasoning to
    LOW energy), so a selector picks ``argmin``.  ``checkpoint`` may be an
    ``epoch_N`` directory or the parent ``out_dir`` (we resolve the latest epoch).
    The base used is read back from the checkpoint's ``train_metrics.json`` so the
    scorer reconstructs the exact stack the adapter was trained on.
    """
    import torch

    ckpt = Path(checkpoint)
    epoch_dir = ckpt if (ckpt / "train_metrics.json").exists() else latest_epoch_checkpoint(ckpt)
    if epoch_dir is None:
        raise RuntimeError(f"no trained checkpoint under {ckpt.as_posix()}")
    metrics = _read_json(epoch_dir / "train_metrics.json")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"unreadable train_metrics.json in {epoch_dir.as_posix()}")
    specs = metrics.get("model_specs", {})
    repo_id = str(specs.get("base_model", ""))
    local_path = str(specs.get("base_cache_path", ""))
    if not local_path:
        raise RuntimeError("checkpoint metrics missing base_cache_path")
    if load_in_4bit is None:
        load_in_4bit = auto_load_in_4bit()

    tokenizer, model = _load_base_stack(
        repo_id,
        local_path,
        load_in_4bit=load_in_4bit,
        device_index=device_index,
        adapter_path=epoch_dir,
        trainable=False,
    )
    model.eval()
    device = next(model.parameters()).device

    energies: list[float] = []
    for text in candidate_texts:
        encoded = tokenizer(
            str(text),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            energy = model(**encoded).logits.float().view(-1)[0]
        energies.append(float(energy.detach().cpu()))
    return energies
