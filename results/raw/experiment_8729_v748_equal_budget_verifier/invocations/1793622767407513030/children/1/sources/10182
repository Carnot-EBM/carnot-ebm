#!/usr/bin/env python3
"""Sudoku energy-as-generator v3 — LATENT (DBAE) compression + energy in the latent.

Human-seeded thesis (operator 2026-06-05). See
docs/research-notes/energy-as-generator-sudoku-thesis.md + memory
project_energy_as_generator_sudoku.

v1: contrastive EBT on the RAW grid = perfect SCORER, failed GENERATOR (decode
drifted into low-energy flatlands). v2: GLOBAL/PCD negatives carved the volume
(flatland -> CARVED) but solve still 0 — the carved landscape had the right minimum
but no findable BASIN from a blank start. Diagnosis (Kona "energy compression"):
need the SECOND compression — DIMENSIONAL/LATENT. Compress the 729-dim grid space
(81 cells x 9) into a compact latent z; descend the energy IN z (a small, findable
space). The decoder maps z to (near-)valid solutions, so descent + energy both
become tractable.

v3 architecture (DBAE-EBM):
  Stage 1 — AUTOENCODER on SOLUTIONS: enc(solution)->z (d_latent, global), dec(z)->
    solution. Pure AE, no puzzle: z is a complete compressed code for the solution.
    Sanity gate: test-set reconstruction accuracy (does z generalize to unseen
    solutions? if low, the latent ceiling is limited — informative either way).
  Stage 2 — LATENT ENERGY (AE FROZEN): E(z, puzzle) low iff dec(z) is consistent
    with the puzzle. Contrastive with GLOBAL negatives via Langevin IN z (PCD) +
    anti-collapse reg. Positives = enc(gold solution).
  Inference — LATENT TRACE DESCENT: z = annealed-Langevin descend E(z|puzzle) from
    random z; grid = argmax dec(z), givens enforced; exact-81 match.

INSTRUMENTED: AE test-recon acc; energy separation; solve-rate; decoded-vs-gold
latent energy (flatland diagnostic in z). VERDICT: solve-rate climbs off v2's 0.000
toward refiner 0.182 => LATENT compression is the missing piece (validates Kona's
full framing + Carnot DBAE). Still ~0 => the wall is deeper (AE generalization, or
the energy/decoder capacity).

GPU: INTERNAL RTX 3090 only. Conductor PAUSED.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sudoku_energy_vs_ar_v1 import (
    load_split, gold_digits, sudoku_violations,
    VOCAB, SEQ, BLANK, DIGIT0, N_DIGITS,
)


def enc_block(hidden, heads):
    return nn.TransformerEncoderLayer(hidden, heads, hidden * 4, batch_first=True,
                                      activation="gelu", dropout=0.0, norm_first=True)


class AutoEncoder(nn.Module):
    """enc(solution digits 0..8) -> z (d_latent global); dec(z) -> 81x9 logits."""
    def __init__(self, hidden=192, layers=3, heads=4, d_latent=256):
        super().__init__()
        self.dig = nn.Embedding(N_DIGITS, hidden)
        self.epos = nn.Embedding(SEQ, hidden)
        self.enc = nn.TransformerEncoder(enc_block(hidden, heads), layers)
        self.to_z = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, d_latent))
        self.from_z = nn.Linear(d_latent, hidden)
        self.dpos = nn.Embedding(SEQ, hidden)
        self.dec = nn.TransformerEncoder(enc_block(hidden, heads), layers)
        self.head = nn.Linear(hidden, N_DIGITS)

    def encode(self, yd):  # yd: (B,81) digits 0..8
        h = self.dig(yd) + self.epos(torch.arange(SEQ, device=yd.device))[None]
        h = self.enc(h).mean(1)
        return self.to_z(h)                                   # (B,d_latent)

    def decode(self, z):   # z: (B,d_latent) -> (B,81,9)
        h = self.dpos(torch.arange(SEQ, device=z.device))[None] + self.from_z(z)[:, None]
        h = self.dec(h)
        return self.head(h)


class LatentEnergy(nn.Module):
    """E(z, puzzle) -> scalar."""
    def __init__(self, hidden=192, d_latent=256):
        super().__init__()
        self.ptok = nn.Embedding(VOCAB, hidden)
        self.ppos = nn.Embedding(SEQ, hidden)
        self.penc = nn.TransformerEncoder(enc_block(hidden, 4), 2)
        self.mlp = nn.Sequential(
            nn.Linear(d_latent + hidden, hidden * 2), nn.GELU(),
            nn.Linear(hidden * 2, hidden), nn.GELU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def puzzle_summary(self, x):
        h = self.ptok(x) + self.ppos(torch.arange(SEQ, device=x.device))[None]
        return self.penc(h).mean(1)                           # (B,hidden)

    def forward(self, z, psum):
        return self.mlp(torch.cat([z, psum], -1)).squeeze(-1)  # (B,)


# ----------------------------------------------------------------- stage 1: AE
def train_ae(ae, xtr, ytr, xev, yev, args, device):
    opt = torch.optim.AdamW(ae.parameters(), lr=args.ae_lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.ae_lr, total_steps=args.ae_steps, pct_start=0.1)
    n = xtr.shape[0]; t0 = time.time(); ae.train()
    for step in range(args.ae_steps):
        idx = torch.randint(0, n, (args.batch,))
        yd = gold_digits(ytr[idx].to(device))
        logits = ae.decode(ae.encode(yd))
        loss = F.cross_entropy(logits.reshape(-1, N_DIGITS), yd.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0); opt.step(); sched.step()
        if step % args.log_every == 0 or step == args.ae_steps - 1:
            print(f"[ae] step={step} loss={loss.item():.4f} t={time.time()-t0:.0f}s", flush=True)
    # sanity: test-set reconstruction accuracy (does z generalise to unseen solutions?)
    ae.eval()
    with torch.no_grad():
        rec_exact = rec_cell = nn_ = 0
        for i in range(0, xev.shape[0], 256):
            yd = gold_digits(yev[i:i + 256].to(device))
            rec = ae.decode(ae.encode(yd)).argmax(-1)
            rec_exact += (rec == yd).all(-1).sum().item()
            rec_cell += (rec == yd).float().mean(-1).sum().item()
            nn_ += yd.shape[0]
    return {"ae_test_recon_exact": rec_exact / nn_, "ae_test_recon_cell": rec_cell / nn_,
            "ae_train_seconds": time.time() - t0}


# ------------------------------------------------ stage 2: latent energy (CD)
def langevin_z(energy, z, psum, steps, step_size, noise):
    for _ in range(steps):
        z = z.detach().requires_grad_(True)
        e = energy(z, psum).sum()
        (g,) = torch.autograd.grad(e, z)
        z = z - step_size * g + noise * torch.randn_like(z)
    return z.detach()


def train_energy(ae, energy, xtr, ytr, args, device):
    for p in ae.parameters():
        p.requires_grad_(False)
    ae.eval()
    opt = torch.optim.AdamW(energy.parameters(), lr=args.e_lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.e_lr, total_steps=args.e_steps, pct_start=0.1)
    n = xtr.shape[0]; t0 = time.time(); energy.train()
    d_latent = args.d_latent
    buf = torch.randn(args.buffer, d_latent, device=device)
    for step in range(args.e_steps):
        idx = torch.randint(0, n, (args.batch,))
        xb = xtr[idx].to(device); yd = gold_digits(ytr[idx].to(device))
        with torch.no_grad():
            z_pos = ae.encode(yd)                              # (B,d_latent)
            psum = energy.puzzle_summary(xb)
        bsel = torch.randint(0, args.buffer, (args.batch,), device=device)
        z0 = buf[bsel].clone()
        reinit = (torch.rand(args.batch, device=device) < args.reinit_frac)
        z0[reinit] = torch.randn_like(z0[reinit])
        with torch.enable_grad():
            z_neg = langevin_z(energy, z0, psum, args.k_langevin, args.lang_step, args.lang_noise)
        buf[bsel] = z_neg
        e_pos = energy(z_pos, psum); e_neg = energy(z_neg, psum)
        cd = e_pos.mean() - e_neg.mean()
        reg = args.alpha * (e_pos ** 2 + e_neg ** 2).mean()
        loss = cd + reg
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(energy.parameters(), 1.0); opt.step(); sched.step()
        if step % args.log_every == 0 or step == args.e_steps - 1:
            print(f"[lat-e] step={step} loss={loss.item():.4f} cd={cd.item():.4f} "
                  f"e_pos={e_pos.mean().item():.3f} e_neg={e_neg.mean().item():.3f} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    return {"energy_train_seconds": time.time() - t0}


# ------------------------------------------------------------- inference / eval
def decode_latent(ae, energy, xb, args):
    """Annealed-Langevin descent in z; decode; enforce givens."""
    B = xb.shape[0]
    psum = energy.puzzle_summary(xb)
    z = torch.randn(B, args.d_latent, device=xb.device)
    n = args.decode_steps
    for i in range(n):
        noise = args.decode_noise0 * (1 - i / max(1, n - 1))
        with torch.enable_grad():
            z = langevin_z(energy, z, psum, 1, args.decode_step, noise)
    grid = ae.decode(z).argmax(-1)
    blanks = (xb == BLANK)
    grid = torch.where(blanks, grid, gold_digits(xb).clamp(0, 8))  # enforce givens
    return grid, z, psum


def evaluate(ae, energy, xev, yev, args, device, bs=256):
    ae.eval(); energy.eval()
    n = xev.shape[0]; hits = 0; blank_c = 0.0; blank_t = 0.0; viol = 0
    e_dec = 0.0; e_gold = 0.0; seen = 0
    for i in range(0, n, bs):
        xb = xev[i:i + bs].to(device); yd = gold_digits(yev[i:i + bs].to(device))
        with torch.enable_grad():
            grid, z_dec, psum = decode_latent(ae, energy, xb, args)
        blanks = (xb == BLANK); blank_t += blanks.float().sum().item()
        hits += (grid == yd).all(-1).sum().item()
        blank_c += ((grid == yd) & blanks).float().sum().item()
        viol += sudoku_violations(grid).sum().item()
        with torch.no_grad():
            e_dec += energy(z_dec, psum).sum().item()
            e_gold += energy(ae.encode(yd), psum).sum().item()
        seen += xb.shape[0]
    return {
        "latent_solve_rate": hits / n,
        "latent_blank_cell_acc": blank_c / max(1, blank_t),
        "latent_mean_violations": viol / n,
        "decoded_latent_energy": e_dec / seen,
        "gold_latent_energy": e_gold / seen,
        "flatland_diagnostic": ("CARVED: decoded-z energy ABOVE gold-z"
                                if e_dec > e_gold else
                                "FLATLAND: decoded-z energy BELOW/equal gold-z"),
        "n_eval": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae_steps", type=int, default=8000)
    ap.add_argument("--e_steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d_latent", type=int, default=256)
    ap.add_argument("--ae_lr", type=float, default=3e-4)
    ap.add_argument("--e_lr", type=float, default=2e-4)
    ap.add_argument("--train_n", type=int, default=400000)
    ap.add_argument("--eval_n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--k_langevin", type=int, default=20)
    ap.add_argument("--lang_step", type=float, default=1.0)
    ap.add_argument("--lang_noise", type=float, default=0.1)
    ap.add_argument("--buffer", type=int, default=10000)
    ap.add_argument("--reinit_frac", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--decode_steps", type=int, default=400)
    ap.add_argument("--decode_step", type=float, default=1.0)
    ap.add_argument("--decode_noise0", type=float, default=0.5)
    ap.add_argument("--out", default="results/sudoku_latent_energy_v3.json")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} cuda_devices={torch.cuda.device_count()} d_latent={args.d_latent}", flush=True)
    assert device == "cuda", "refusing to run on CPU — pin the internal 3090"

    xtr, ytr = load_split("train", args.train_n, args.seed)
    xev, yev = load_split("test", args.eval_n, args.seed + 1)
    ae = AutoEncoder(args.hidden, args.layers, args.heads, args.d_latent).to(device)
    energy = LatentEnergy(args.hidden, args.d_latent).to(device)

    m = {"model": "latent_energy_v3", "d_latent": args.d_latent,
         "n_params_ae": sum(p.numel() for p in ae.parameters()),
         "n_params_energy": sum(p.numel() for p in energy.parameters())}
    m.update(train_ae(ae, xtr, ytr, xev, yev, args, device))
    print(f"[ae] test recon: exact={m['ae_test_recon_exact']:.3f} cell={m['ae_test_recon_cell']:.3f}", flush=True)
    m.update(train_energy(ae, energy, xtr, ytr, args, device))
    m.update(evaluate(ae, energy, xev, yev, args, device))
    m["args"] = vars(args); m["gpu"] = torch.cuda.get_device_name(0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in m.items() if k != "args"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
