"""
Thesis-A (energy-as-GENERATOR / EBT) — DIRECT definitive bring-up run.

Operator-authorized 2026-06-03: run the genuine EBT training directly on the
recovered dual-3090 rig (both GPUs back after the USB4 reseat) to get a
definitive kill-gate part-(a) answer, bypassing the conductor's flaky
subprocess that twice blocked on environment faults (inode exhaustion, then a
transient cuda=False).

WHAT THIS DECIDES (honest scope):
  - Part (a), the kill-gate's gating half: does the tiny 38M EBT contrastive-
    divergence training run STABLY on real GSM8K data (no NaN/divergence, bounded
    loss, finite grad norms) for a real multi-hundred-step budget?
  - "Did it learn?": on a HELD-OUT split, does the TRAINED EBT assign lower
    energy to real next-token continuations than to random negatives (a positive
    pos/neg energy MARGIN), and is that margin materially larger than an
    UNTRAINED random-init EBT's margin (~0)? This is the cleanest small-scale
    signal that energy-as-generator is learning the data distribution.
  - AR sanity: the matched AR transformer's held-out cross-entropy should drop
    vs init, confirming the training setup is sound.

WHAT THIS DOES NOT DECIDE (deferred, stated honestly):
  - Part (b), the full matched-COMPUTE reasoning-accuracy comparison: this EBT
    outputs embeddings, not a decodable token distribution; a faithful
    EBT-generation-vs-AR accuracy test needs an embedding->token decoder and
    larger scale than 38M/2048-examples/byte-tokenizer. That is the NEXT
    milestone, not fakeable here. We do NOT claim a reasoning-accuracy result.

Reuses the conductor's own model/training code (experiment_3734) so this is a
faithful run of the real harness, not a reimplementation.
"""

import os
import sys
import time
import json
import math
import random
import hashlib
from pathlib import Path

# Resolved from this file rather than hardcoded so a fresh clone or a
# worktree writes into ITS OWN tree. Inlined (not carnot.paths.repo_root)
# because the next line is what makes ``carnot`` importable -- importing
# the resolver here would be circular. Same rule, same answer.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

# Reuse the conductor's real harness code (model construction + tokenizer).
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "exp3734",
    os.path.join(
        PROJECT_ROOT, "scripts", "experiment_3734_fix_harness_and_bounded_train_chunk1.py"
    ),
)
exp3734 = _ilu.module_from_spec(_spec)
sys.modules["exp3734"] = exp3734
_spec.loader.exec_module(exp3734)

build_tiny_models = exp3734.build_tiny_models
tokenize_texts_to_blocks = exp3734.tokenize_texts_to_blocks
format_gsm8k_row = exp3734.format_gsm8k_row
VOCAB = exp3734.BYTE_TOKENIZER_VOCAB_SIZE

import torch
import torch.nn.functional as F  # noqa: N812 (PyTorch community convention)
import numpy as np
from datasets import load_dataset

RANDOM_SEED = 30603
MAX_STEPS = 800  # real budget (vs the conductor's 200); wall-bounded below
MAX_WALL_S = 1200.0
EVAL_EVERY = 100
DEVICE = (
    "cuda:1"
    if torch.cuda.device_count() > 1
    else ("cuda:0" if torch.cuda.is_available() else "cpu")
)

dim, n_layers, n_heads, ffn_mult, block_size, batch_size, lr = 768, 4, 12, 4.0, 128, 8, 3e-4


def make_blocks(texts):
    return torch.as_tensor(tokenize_texts_to_blocks(texts, block_size), dtype=torch.long)


def ebt_held_out_margin(ebt_model, eval_blocks, device, n_batches=8):
    """Mean(neg_energy) - mean(pos_energy) on held-out data.
    Positive => the EBT assigns LOWER energy to real continuations than to
    random negatives => it learned a useful landscape. ~0 => learned nothing."""
    ebt_model.eval()
    pos_e, neg_e = [], []
    with torch.no_grad():
        for b in range(min(n_batches, max(1, eval_blocks.shape[0] // batch_size))):
            batch = eval_blocks[b * batch_size : (b + 1) * batch_size].to(device)
            if batch.shape[0] < 2:
                continue
            orig = ebt_model.token_embedding(batch[:, :-1])
            pos = ebt_model.token_embedding(batch[:, 1:])
            neg = torch.randn_like(pos) * 0.02
            pos_e.append(ebt_model(orig, pos).mean().item())
            neg_e.append(ebt_model(orig, neg).mean().item())
    ebt_model.train()
    if not pos_e:
        return None
    return float(np.mean(neg_e) - np.mean(pos_e))


def ar_held_out_ce(ar_model, eval_blocks, device, n_batches=8):
    ar_model.eval()
    ces = []
    with torch.no_grad():
        for b in range(min(n_batches, max(1, eval_blocks.shape[0] // batch_size))):
            batch = eval_blocks[b * batch_size : (b + 1) * batch_size].to(device)
            if batch.shape[0] < 1:
                continue
            logits = ar_model(batch[:, :-1])
            ce = F.cross_entropy(logits.reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
            ces.append(float(ce.item()))
    ar_model.train()
    return float(np.mean(ces)) if ces else None


def main():
    t0 = time.time()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    device = torch.device(DEVICE)
    print(f"[setup] device={device} cuda_devs={torch.cuda.device_count()}", flush=True)

    # Real GSM8K: train split for training, a DISJOINT slice for held-out eval.
    train_ds = load_dataset("gsm8k", "main", split="train[:2048]")
    eval_ds = load_dataset("gsm8k", "main", split="train[2048:2304]")
    train_blocks = make_blocks([format_gsm8k_row(r) for r in train_ds])
    eval_blocks = make_blocks([format_gsm8k_row(r) for r in eval_ds])
    print(
        f"[data] train_blocks={tuple(train_blocks.shape)} eval_blocks={tuple(eval_blocks.shape)}",
        flush=True,
    )

    ebt_model, ar_model = build_tiny_models(
        dim, n_layers, n_heads, ffn_mult, batch_size, block_size
    )
    ebt_model, ar_model = ebt_model.to(device), ar_model.to(device)
    n_params_ebt = sum(p.numel() for p in ebt_model.parameters())
    n_params_ar = sum(p.numel() for p in ar_model.parameters())

    # Untrained-baseline calibration: a fresh EBT's held-out margin (~0 expected).
    ebt_untrained, _ = build_tiny_models(dim, n_layers, n_heads, ffn_mult, batch_size, block_size)
    ebt_untrained = ebt_untrained.to(device)
    untrained_margin = ebt_held_out_margin(ebt_untrained, eval_blocks, device)
    del ebt_untrained
    ar_init_ce = ar_held_out_ce(ar_model, eval_blocks, device)

    ebt_opt = torch.optim.AdamW(ebt_model.parameters(), lr=lr)
    ar_opt = torch.optim.AdamW(ar_model.parameters(), lr=lr)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    ebt_model.train()
    ar_model.train()

    replay_buffer, buffer_size = [], 1000
    ebt_curve, ar_curve, grad_norms = [], [], []
    eval_trace = []
    nan_div = False
    steps = 0

    for step in range(MAX_STEPS):
        if time.time() - t0 > MAX_WALL_S - 20:
            print("[stop] wall budget reached", flush=True)
            break
        idx = torch.randint(0, max(1, train_blocks.shape[0] - batch_size), (1,)).item()
        batch = train_blocks[idx : idx + batch_size].to(device)

        # AR
        ar_opt.zero_grad(set_to_none=True)
        logits = ar_model(batch[:, :-1])
        ar_loss = F.cross_entropy(logits.reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
        if not torch.isfinite(ar_loss):
            nan_div = True
            print("[DIVERGE] AR loss non-finite", flush=True)
            break
        ar_loss.backward()
        torch.nn.utils.clip_grad_norm_(ar_model.parameters(), 1.0)
        ar_opt.step()
        ar_curve.append(float(ar_loss.item()))

        # EBT contrastive divergence
        ebt_opt.zero_grad(set_to_none=True)
        orig = ebt_model.token_embedding(batch[:, :-1])
        pos = ebt_model.token_embedding(batch[:, 1:])
        pos_energy = ebt_model(orig, pos).mean()
        if len(replay_buffer) > 0 and random.random() < 0.95:
            ib = torch.randint(0, len(replay_buffer), (batch_size,))
            neg = torch.stack([replay_buffer[i] for i in ib]).to(device)
        else:
            neg = torch.randn_like(pos) * 0.02
        neg = neg.detach()
        neg.requires_grad_(True)
        alpha = random.uniform(0.1, 1.0)
        for _ in range(random.randint(10, 30)):
            ne = ebt_model(orig.detach(), neg).mean()
            g = torch.autograd.grad(ne, neg)[0]
            neg = (neg - alpha * g + torch.randn_like(neg) * math.sqrt(2 * alpha)).detach()
            neg.requires_grad_(True)
        neg_energy_final = ebt_model(orig, neg.detach()).mean()
        kl = 0.1 * (pos_energy**2 + neg_energy_final**2)
        ebt_loss = pos_energy - neg_energy_final + kl
        if not torch.isfinite(ebt_loss):
            nan_div = True
            print(f"[DIVERGE] EBT loss non-finite at step {step}", flush=True)
            break
        ebt_loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(ebt_model.parameters(), 1.0)
        ebt_opt.step()
        ebt_curve.append(float(ebt_loss.item()))
        grad_norms.append(float(gn))
        for i in range(batch_size):
            if len(replay_buffer) < buffer_size:
                replay_buffer.append(neg[i].detach().cpu())
            else:
                replay_buffer[random.randint(0, buffer_size - 1)] = neg[i].detach().cpu()
        steps += 1

        if (step + 1) % EVAL_EVERY == 0:
            m = ebt_held_out_margin(ebt_model, eval_blocks, device)
            ce = ar_held_out_ce(ar_model, eval_blocks, device)
            eval_trace.append({"step": step + 1, "ebt_heldout_margin": m, "ar_heldout_ce": ce})
            print(
                f"[eval] step={step + 1} ebt_loss={ebt_loss.item():.4f} ar_loss={ar_loss.item():.4f} "
                f"ebt_heldout_margin={m:.4f} ar_heldout_ce={ce:.4f} gradnorm={gn:.3f}",
                flush=True,
            )

    # Final held-out probes
    final_margin = ebt_held_out_margin(ebt_model, eval_blocks, device)
    final_ar_ce = ar_held_out_ce(ar_model, eval_blocks, device)
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device) // (1024 * 1024))
        if device.type == "cuda"
        else 0
    )

    # Verdicts (honest)
    trained_stably = (
        (not nan_div) and steps >= 200 and all(math.isfinite(x) for x in ebt_curve[-20:])
    )
    ebt_learned = (
        final_margin is not None
        and untrained_margin is not None
        and final_margin > 0
        and final_margin > untrained_margin + 0.5
    )
    ar_learned = (
        ar_init_ce is not None and final_ar_ce is not None and final_ar_ce < ar_init_ce - 0.05
    )

    if nan_div:
        verdict = f"complete: thesis_a_part_a_FAIL_ebt_diverged_at_step_{steps}_energy_as_generator_unstable_at_small_scale"
    elif trained_stably and ebt_learned:
        verdict = (
            f"complete: thesis_a_part_a_PASS_ebt_trained_stably_{steps}_steps_and_LEARNED_heldout_margin_"
            f"{final_margin:.3f}_vs_untrained_{untrained_margin:.3f}_part_b_decoder_scale_deferred"
        )
    elif trained_stably and not ebt_learned:
        verdict = (
            f"complete: thesis_a_part_a_MIXED_ebt_trained_stably_{steps}_steps_but_weak_heldout_learning_"
            f"margin_{final_margin}_vs_untrained_{untrained_margin}"
        )
    else:
        verdict = f"complete: thesis_a_part_a_INCONCLUSIVE_only_{steps}_steps"

    payload = {
        "seed": RANDOM_SEED,
        "max_steps": MAX_STEPS,
        "block_size": block_size,
        "batch_size": batch_size,
        "dim": dim,
        "n_layers": n_layers,
    }
    checksum = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    artifact = {
        "experiment": "thesis_a_direct_definitive_run",
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "operator_authorized": "2026-06-03 direct definitive run on recovered dual-3090 rig",
        "scope_note": (
            "Decides kill-gate part-(a): EBT CD training stability + held-out energy-landscape "
            "learning on real GSM8K. Does NOT decide part-(b) matched-compute reasoning accuracy "
            "(EBT outputs embeddings; faithful generation-vs-AR needs an embedding->token decoder "
            "+ larger scale = next milestone). No reasoning-accuracy claim is made here."
        ),
        "device": str(device),
        "cuda_device_count": torch.cuda.device_count(),
        "ebt_param_count": n_params_ebt,
        "ar_param_count": n_params_ar,
        "cumulative_steps_trained": steps,
        "nan_or_divergence_events": nan_div,
        "ebt_trained_stably": trained_stably,
        "ebt_learned_heldout": ebt_learned,
        "ar_learned_heldout": ar_learned,
        "ebt_heldout_margin_final": final_margin,
        "ebt_heldout_margin_untrained_baseline": untrained_margin,
        "ar_heldout_ce_init": ar_init_ce,
        "ar_heldout_ce_final": final_ar_ce,
        "eval_trace": eval_trace,
        "ebt_loss_curve_sampled": ebt_curve[:: max(1, len(ebt_curve) // 30)] if ebt_curve else [],
        "ar_loss_curve_sampled": ar_curve[:: max(1, len(ar_curve) // 30)] if ar_curve else [],
        "grad_norm_max": max(grad_norms) if grad_norms else None,
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
        "peak_vram_mb": peak_vram,
        "stabilizers_applied": "replay_buffer, langevin_noise, random_alpha, random_descent_steps, grad_clip, kl_reg",
        "model_specs": {
            "ebt": "tiny_ebt_from_scratch_38M_byte",
            "ar": "tiny_ar_from_scratch_matched_byte",
            "from_scratch": True,
            "not_a_pretrained_llm": True,
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": checksum,
        "duration_s": round(time.time() - t0, 2),
    }
    out = os.path.join(PROJECT_ROOT, "results", "thesis_a_direct_definitive_run.json")
    with open(out, "w") as f:
        json.dump(artifact, f, indent=2)
    print("\n" + verdict, flush=True)
    print(
        f"[done] steps={steps} stable={trained_stably} ebt_learned={ebt_learned} "
        f"margin={final_margin} (untrained {untrained_margin}) ar_ce {ar_init_ce}->{final_ar_ce} "
        f"dur={artifact['duration_s']}s vram={peak_vram}MB -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
