#!/usr/bin/env python3
"""Sudoku amortized-inference CAPSTONE — confirm the v1->v3 conclusion by REVERSAL.

v1->v3 showed: energy SCORES Sudoku perfectly but energy-DESCENT generates nothing
(0%) even with a perfect latent + perfect carving — the wall is AMORTIZED INFERENCE
(the learned funnel), not the energy. The refiner (a learned generation map) solves
~18%. This capstone confirms that mechanistically by adding the learned funnel back
to the energy and showing solve-rate JUMPS:

  trains a Refiner (the amortized generator) + a global-negative carved EBT (the
  scorer), then compares four inference procedures with the SAME trained models:
    (1) random-init energy descent  -> expect ~0 (the v1->v3 failure, recomputed)
    (2) refiner greedy              -> the ~18% baseline (amortized alone)
    (3) refiner-init energy descent -> energy polishing from the learned funnel
    (4) energy-reranked refiner@K   -> verifier picks best of K refiner samples

If (3)/(4) >> (1), the wall was the funnel (amortization), not the energy — and the
energy ADDS value as a SCORER on the amortized generator (the Carnot verifier+
generator product thesis). GPU: INTERNAL RTX 3090 only. Conductor PAUSED.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sudoku_energy_vs_ar_v1 import (
    load_split, gold_digits, sudoku_violations, EBT, Refiner,
    SEQ, BLANK, DIGIT0, N_DIGITS,
)
from sudoku_energy_compression_v2 import clamp_givens, langevin


def train_refiner_model(xtr, ytr, args, device):
    model = Refiner(args.hidden, max(2, args.layers // 2), args.heads, args.n_cycles).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.r_steps, pct_start=0.1)
    n = xtr.shape[0]; t0 = time.time(); model.train()
    for step in range(args.r_steps):
        idx = torch.randint(0, n, (args.batch,))
        xb = xtr[idx].to(device); yd = gold_digits(ytr[idx].to(device))
        outs = model(xb, deep_supervision=True)
        loss = sum(F.cross_entropy(o.reshape(-1, N_DIGITS), yd.reshape(-1)) for o in outs) / len(outs)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % args.log_every == 0 or step == args.r_steps - 1:
            print(f"[refiner] step={step} loss={loss.item():.4f} t={time.time()-t0:.0f}s", flush=True)
    return model


def train_ebt_global(xtr, ytr, args, device):
    """EBT with GLOBAL Langevin/PCD negatives (carved energy), v2-style."""
    model = EBT(args.hidden, args.layers, args.heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.e_lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.e_lr, total_steps=args.e_steps, pct_start=0.1)
    n = xtr.shape[0]; t0 = time.time(); model.train()
    buf = torch.randn(args.buffer, SEQ, N_DIGITS, device=device)
    for step in range(args.e_steps):
        idx = torch.randint(0, n, (args.ebatch,))
        xb = xtr[idx].to(device); yd = gold_digits(ytr[idx].to(device))
        bsel = torch.randint(0, args.buffer, (args.ebatch,), device=device)
        L0 = buf[bsel].clone()
        reinit = (torch.rand(args.ebatch, device=device) < 0.05)
        L0[reinit] = torch.randn_like(L0[reinit])
        with torch.enable_grad():
            Lneg = langevin(model, xb, L0, args.k_langevin, 1.0, 0.1)
        buf[bsel] = Lneg
        gold_oh = F.one_hot(yd, N_DIGITS).float()
        e_gold = model(xb, clamp_givens(gold_oh, xb))
        e_neg = model(xb, clamp_givens(F.softmax(Lneg, -1), xb))
        loss = (e_gold.mean() - e_neg.mean()) + 0.1 * (e_gold ** 2 + e_neg ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % args.log_every == 0 or step == args.e_steps - 1:
            print(f"[ebt] step={step} loss={loss.item():.4f} e_gold={e_gold.mean().item():.2f} "
                  f"e_neg={e_neg.mean().item():.2f} t={time.time()-t0:.0f}s", flush=True)
    return model


def solve(grid, yd):
    return (grid == yd).all(-1).float().mean().item()


@torch.no_grad()
def refiner_greedy(refiner, xb):
    return refiner(xb).argmax(-1)


@torch.no_grad()
def refiner_samples(refiner, xb, K, temp):
    logits = refiner(xb)                                       # (B,81,9) final-cycle logits
    B = xb.shape[0]
    out = torch.zeros(B, K, SEQ, dtype=torch.long, device=xb.device)
    for k in range(K):
        probs = F.softmax(logits / temp, -1)
        s = torch.multinomial(probs.reshape(-1, N_DIGITS), 1).reshape(B, SEQ)
        blanks = (xb == BLANK)
        out[:, k] = torch.where(blanks, s, gold_digits(xb).clamp(0, 8))
    return out


def energy_descent(ebt, xb, init_logits, steps, lr):
    """Descend EBT energy from init_logits (B,81,9); givens clamped; argmax."""
    L = init_logits.detach().clone().requires_grad_(True)
    blanks = (xb == BLANK)
    opt = torch.optim.Adam([L], lr=lr)
    for _ in range(steps):
        soft = clamp_givens(F.softmax(L, -1), xb)
        e = ebt(xb, soft).mean()
        opt.zero_grad(); e.backward(); opt.step()
    with torch.no_grad():
        return torch.where(blanks, F.softmax(L, -1).argmax(-1), gold_digits(xb).clamp(0, 8))


def evaluate_capstone(refiner, ebt, xev, yev, args, device, bs=128):
    refiner.eval(); ebt.eval()
    n = xev.shape[0]
    acc = {k: 0 for k in ["rand_descent", "refiner_greedy", "refiner_init_descent", "energy_rerank"]}
    for i in range(0, n, bs):
        xb = xev[i:i + bs].to(device); yd = gold_digits(yev[i:i + bs].to(device))
        # (1) random-init energy descent (the v1->v3 failure, recomputed for THIS ebt)
        with torch.enable_grad():
            rand0 = torch.randn(xb.shape[0], SEQ, N_DIGITS, device=device)
            g1 = energy_descent(ebt, xb, rand0, args.descent_steps, args.descent_lr)
        acc["rand_descent"] += (g1 == yd).all(-1).sum().item()
        # (2) refiner greedy (amortized alone)
        g2 = refiner_greedy(refiner, xb)
        acc["refiner_greedy"] += (g2 == yd).all(-1).sum().item()
        # (3) refiner-init energy descent (learned funnel + energy polish)
        with torch.no_grad():
            rlogits = refiner(xb)                              # (B,81,9)
        with torch.enable_grad():
            g3 = energy_descent(ebt, xb, rlogits * args.init_scale, args.descent_steps, args.descent_lr)
        acc["refiner_init_descent"] += (g3 == yd).all(-1).sum().item()
        # (4) energy-reranked refiner@K (verifier picks best of K refiner samples)
        with torch.no_grad():
            samples = refiner_samples(refiner, xb, args.rerank_k, args.rerank_temp)   # (B,K,81)
            energies = torch.stack([ebt(xb, F.one_hot(samples[:, k], N_DIGITS).float())
                                    for k in range(args.rerank_k)], 1)                # (B,K)
            best = samples[torch.arange(xb.shape[0]), energies.argmin(1)]             # (B,81)
        acc["energy_rerank"] += (best == yd).all(-1).sum().item()
    return {f"{k}_solve_rate": v / n for k, v in acc.items()} | {"n_eval": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r_steps", type=int, default=15000)
    ap.add_argument("--e_steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ebatch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n_cycles", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--e_lr", type=float, default=2e-4)
    ap.add_argument("--train_n", type=int, default=400000)
    ap.add_argument("--eval_n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--k_langevin", type=int, default=20)
    ap.add_argument("--buffer", type=int, default=10000)
    ap.add_argument("--descent_steps", type=int, default=200)
    ap.add_argument("--descent_lr", type=float, default=0.3)
    ap.add_argument("--init_scale", type=float, default=5.0)   # sharpen refiner logits as descent init
    ap.add_argument("--rerank_k", type=int, default=32)
    ap.add_argument("--rerank_temp", type=float, default=1.0)
    ap.add_argument("--out", default="results/sudoku_amortized_capstone_v4.json")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} cuda_devices={torch.cuda.device_count()}", flush=True)
    assert device == "cuda", "refusing to run on CPU — pin the internal 3090"

    xtr, ytr = load_split("train", args.train_n, args.seed)
    xev, yev = load_split("test", args.eval_n, args.seed + 1)
    t0 = time.time()
    refiner = train_refiner_model(xtr, ytr, args, device)
    ebt = train_ebt_global(xtr, ytr, args, device)
    m = evaluate_capstone(refiner, ebt, xev, yev, args, device)
    m.update({"model": "amortized_capstone_v4", "seed": args.seed,
              "duration_s": time.time() - t0, "args": vars(args),
              "gpu": torch.cuda.get_device_name(0)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in m.items() if k != "args"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
