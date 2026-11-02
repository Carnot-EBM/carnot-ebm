"""
Thesis-A part (b) SCALED — does energy-as-GENERATOR beat AR at matched compute,
given a BIGGER model + more steps + a LEARNED decoder + a not-memorizable task?

Operator-authorized 2026-06-03 ("scale up part b"). The small-scale part-(b) was
BOUNDED (EBT 0% vs AR 6.2%) but with two confounds a scale-up must remove:
  (1) DECODE bottleneck — the energy landscape is discriminative (part-a margin
      0.72) but the simple energy-argmin decode produced garbage. Fix: a LEARNED
      decoder (emb->token), trained on the EBT's energy-converged embeddings, so
      the EBT gets a fair generation path.
  (2) SCALE — 38M / 2-digit / 3500 steps is tiny. Fix: bigger model (dim x
      n_layers), more steps, and 3-digit addition (1M problems = not memorizable,
      a real generalization test).

This script runs ONE seed on ONE device (so seeds run in parallel across the two
3090s). PAUSE the conductor before running (its gpu_monitor.kill_zombies SIGTERMs
any GPU process it doesn't own).

EBT generation methods compared (each vs its MATCHED-compute AR baseline):
  - EBT-argmin: energy-argmin over vocab (VOCAB evals/token) vs AR@VOCAB self-consistency.
  - EBT-descent+decoder: K-step Langevin energy descent from noise -> learned
    decoder (K evals/token) vs AR@K self-consistency.
PASS (this seed) if EITHER EBT method beats its matched AR baseline AND headroom
holds (AR@1 in (0.05,0.95)). Multi-seed aggregation happens in a separate step.
"""

import os
import sys
import time
import json
import math
import random
import hashlib
import argparse
from collections import Counter
from pathlib import Path

# Resolved from this file rather than hardcoded so a fresh clone or a
# worktree writes into ITS OWN tree. Inlined (not carnot.paths.repo_root)
# because the next line is what makes ``carnot`` importable -- importing
# the resolver here would be circular. Same rule, same answer.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "pb", os.path.join(PROJECT_ROOT, "scripts", "thesis_a_part_b_matched_compute.py")
)
pb = _ilu.module_from_spec(_spec)
sys.modules["pb"] = pb
_spec.loader.exec_module(pb)

enc, dec_ids = pb.enc, pb.dec_ids
build_corpus, corpus_to_blocks = pb.build_corpus, pb.corpus_to_blocks
ar_greedy, ar_selfconsistency, ebt_generate = pb.ar_greedy, pb.ar_selfconsistency, pb.ebt_generate
VOCAB, EOS, OFF = pb.VOCAB, pb.EOS, pb.OFF

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 (PyTorch community convention)
import numpy as np


def build_models(dim, n_layers, n_heads, block_size, device):
    ebt, ar = pb.build_tiny_models(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim_multiplier=4.0,
        batch_size=16,
        block_size=block_size,
    )
    return ebt.to(device), ar.to(device)


def descend(ebt, ctx, device, K, step=0.12):
    """BATCHED K-step Langevin energy descent of the next-token embedding.
    ctx: LongTensor [B, L] (same length L across the batch). Returns [B, dim].
    Batching collapses B*K tiny autograd launches into K, which is far faster AND
    avoids the CUDA launch-failure churn that per-example descent triggered."""
    B, L = ctx.shape
    orig = ebt.token_embedding(ctx).detach()
    known = (
        ebt.token_embedding(ctx[:, 1:]).detach()
        if L >= 2
        else torch.zeros((B, 0, orig.shape[-1]), device=device)
    )
    cand = (torch.randn((B, 1, orig.shape[-1]), device=device) * 0.02).requires_grad_(True)
    for _ in range(K):
        e = ebt(orig, torch.cat([known, cand], dim=1))[
            :, -1, 0
        ].sum()  # sum over batch; grad is per-example
        g = torch.autograd.grad(e, cand)[0]
        cand = (cand - step * g).detach().requires_grad_(True)
    return cand[:, 0].detach()  # [B, dim]


def fit_decoder(ebt, blocks, device, K, steps, bs=64, log=print):
    """Train a decoder emb->VOCAB on the EBT's energy-converged embeddings (the
    fair generation head). Each step samples a FIXED position P (so all contexts
    share length P and the descent is one batched call), descends from noise to the
    EBT's preferred emb, and trains decoder(emb)->CE(true token at P)."""
    dim = ebt.token_embedding.weight.shape[1]
    dec = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, VOCAB)).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-3)
    ebt.eval()
    Lmax = blocks.shape[1]
    for step in range(steps):
        P = random.randint(2, Lmax - 2)  # fixed position -> batchable
        bidx = torch.randint(0, blocks.shape[0], (bs,))
        ctx = blocks[bidx, :P].to(device)  # [bs, P]
        tgt = blocks[bidx, P].to(device)  # [bs]
        emb = descend(ebt, ctx, device, K)  # [bs, dim] (one batched descent)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(dec(emb), tgt)
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            log(f"[decoder] step={step + 1} ce={loss.item():.3f}")
    dec.eval()
    return dec


def ebt_descent_generate(ebt, dec, pid, ans_len, device, K):
    """EBT generation: energy-descend from noise per token (batched B=1), decode
    via the learned decoder. Compute = K energy evals/token. Returns (ids, n_eval)."""
    ids = list(pid)
    nf = 0
    for _ in range(ans_len):
        emb = descend(ebt, torch.tensor([ids], device=device), device, K)
        nf += K
        with torch.no_grad():
            ids.append(int(dec(emb)[0].argmax()))
    return ids[len(pid) :], nf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--digits", type=int, default=3)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--decoder-steps", type=int, default=1500)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.decoder_steps, a.n_eval, a.layers = 2500, 500, 30, 6

    t0 = time.time()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.cuda.manual_seed_all(a.seed)
    blk = 48
    print(
        f"[setup] seed={a.seed} dev={dev} digits={a.digits} dim={a.dim} layers={a.layers} "
        f"steps={a.steps} K={a.K} smoke={a.smoke}",
        flush=True,
    )

    mu = (10**a.digits) ** 2
    n_train = min(40000, int(mu * 0.7))
    train_items = build_corpus(a.digits, n_train, a.seed)
    tp = {t[0] for t in train_items}
    eval_items = build_corpus(a.digits, 4000, a.seed + 777, exclude=tp)
    blocks = corpus_to_blocks(train_items, blk)
    print(
        f"[data] train={len(train_items)} eval={len(eval_items)} blocks={tuple(blocks.shape)}",
        flush=True,
    )

    ebt, ar = build_models(a.dim, a.layers, a.heads, blk, dev)
    n_ebt = sum(p.numel() for p in ebt.parameters())
    nan = pb.train_models(
        ebt, ar, blocks, dev, a.steps, bs=16, langevin=(5, 15), log=lambda m: print(m, flush=True)
    )
    print(f"[train] done nan={nan} ebt_params={n_ebt}", flush=True)

    dec = fit_decoder(ebt, blocks, dev, a.K, a.decoder_steps, log=lambda m: print(m, flush=True))

    # eval
    ans_len = a.digits + 1
    ebt.eval()
    ar.eval()
    items = eval_items[: a.n_eval]
    ar1 = arV = arK = e_argmin = e_dec = 0
    arV_nf = arK_nf = eA_nf = eD_nf = 0
    samples = []
    for j, (p, true_ans) in enumerate(items):
        pid = enc(p)
        true = enc(true_ans)
        g1, _ = ar_greedy(ar, pid, ans_len, dev)
        ar1 += g1 == true
        gV, nf = ar_selfconsistency(ar, pid, ans_len, dev, VOCAB)
        arV_nf += nf
        arV += gV == true
        gK, nf = ar_selfconsistency(ar, pid, ans_len, dev, a.K)
        arK_nf += nf
        arK += gK == true
        ga, nf = ebt_generate(ebt, pid, ans_len, dev)
        eA_nf += nf
        e_argmin += ga == true
        gd, nf = ebt_descent_generate(ebt, dec, pid, ans_len, dev, a.K)
        eD_nf += nf
        e_dec += gd == true
        if j < 8:
            samples.append(
                {
                    "prompt": p,
                    "true": true_ans,
                    "ar1": dec_ids(g1),
                    "ebt_argmin": dec_ids(ga),
                    "ebt_descent_dec": dec_ids(gd),
                }
            )
        if (j + 1) % 20 == 0:
            print(
                f"[eval] {j + 1}/{len(items)} ar1={ar1 / (j + 1):.3f} arV={arV / (j + 1):.3f} "
                f"arK={arK / (j + 1):.3f} eArgmin={e_argmin / (j + 1):.3f} eDec={e_dec / (j + 1):.3f}",
                flush=True,
            )
    n = len(items)
    R = {
        k: v / n
        for k, v in [
            ("ar1", ar1),
            ("arV", arV),
            ("arK", arK),
            ("ebt_argmin", e_argmin),
            ("ebt_descent_dec", e_dec),
        ]
    }
    headroom = 0.05 < R["ar1"] < 0.95
    best_ebt = max(R["ebt_argmin"], R["ebt_descent_dec"])
    matched_ar = R["arV"] if R["ebt_argmin"] >= R["ebt_descent_dec"] else R["arK"]
    if not headroom:
        verdict = f"complete: thesis_a_part_b_scaled_REJECTED_no_headroom_ar1_{R['ar1']:.3f}"
    elif nan:
        verdict = "complete: thesis_a_part_b_scaled_INCONCLUSIVE_diverged"
    elif best_ebt > matched_ar:
        verdict = f"complete: thesis_a_part_b_scaled_PASS_seed{a.seed}_ebt_{best_ebt:.3f}_beats_ar_{matched_ar:.3f}"
    else:
        verdict = (
            f"complete: thesis_a_part_b_scaled_BOUNDED_seed{a.seed}_ebt_{best_ebt:.3f}_le_ar_"
            f"{matched_ar:.3f}_ar1_{R['ar1']:.3f}_headroom_ok"
        )

    art = {
        "experiment": f"thesis_a_part_b_scaled_seed{a.seed}",
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "operator_authorized": "2026-06-03 scale-up part-b (bigger model + learned decoder + 3-digit + multi-seed)",
        "task": f"fixed-width {a.digits}-digit addition MSD-first (AR-hostile), held-out split",
        "smoke": a.smoke,
        "training_diverged": nan,
        "headroom_ok": headroom,
        "model_specs": {
            "dim": a.dim,
            "n_layers": a.layers,
            "ebt_params": n_ebt,
            "ebt_decoder": "learned emb->token MLP on energy-converged embeddings",
            "from_scratch": True,
        },
        "ar1_greedy_acc": R["ar1"],
        "arV_selfconsistency_acc": R["arV"],
        "arK_selfconsistency_acc": R["arK"],
        "ebt_argmin_acc": R["ebt_argmin"],
        "ebt_descent_decoder_acc": R["ebt_descent_dec"],
        "best_ebt_acc": best_ebt,
        "matched_ar_acc": matched_ar,
        "delta_best_ebt_minus_matched_ar": best_ebt - matched_ar,
        "matched_compute": {
            "ebt_argmin_evals": eA_nf,
            "arV_forward": arV_nf,
            "ebt_descent_evals": eD_nf,
            "arK_forward": arK_nf,
            "argmin_ratio": (eA_nf / arV_nf) if arV_nf else None,
            "descent_ratio": (eD_nf / arK_nf) if arK_nf else None,
            "K": a.K,
        },
        "n_eval": n,
        "ebt_sample_outputs": samples,
        "device": str(dev),
        "random_seed": a.seed,
        "reproducibility_checksum": hashlib.sha256(
            json.dumps(
                {
                    "seed": a.seed,
                    "digits": a.digits,
                    "dim": a.dim,
                    "layers": a.layers,
                    "steps": a.steps,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "duration_s": round(time.time() - t0, 2),
    }
    out = os.path.join(
        PROJECT_ROOT,
        "results",
        f"thesis_a_part_b_scaled_seed{a.seed}{'_smoke' if a.smoke else ''}.json",
    )
    with open(out, "w") as f:
        json.dump(art, f, indent=2)
    print("\n" + verdict, flush=True)
    print(
        f"[done] ar1={R['ar1']:.3f} arV={R['arV']:.3f} arK={R['arK']:.3f} "
        f"ebt_argmin={R['ebt_argmin']:.3f} ebt_dec={R['ebt_descent_dec']:.3f} "
        f"headroom={headroom} dur={art['duration_s']}s -> {out}",
        flush=True,
    )
    print("[samples] " + json.dumps(samples[:5]), flush=True)


if __name__ == "__main__":
    main()
