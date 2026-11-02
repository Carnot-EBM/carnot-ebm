#!/usr/bin/env python3
"""Energy-as-generator vs autoregression on Sudoku-Extreme (regime-corrected).

Human-seeded thesis (operator 2026-06-05) — see
docs/research-notes/energy-as-generator-sudoku-thesis.md.

WHY: Carnot's P1/Thesis-A tested energy-as-generator on ARITHMETIC and found it
bounded — but that is the WRONG regime (self-consistency is near-ceiling on
addition, no headroom). The Kona EBM Sudoku demo (logicalintelligence.com) claims
energy-descent crushes autoregression on constraint grids (96% vs 2%). This harness
tests that claim on Sudoku-Extreme at MATCHED COMPUTE, with an honest AR control.

This file builds three generators (selected by --model): an autoregressive
transformer (also the HEADROOM positive control), an energy-based model (EBT,
energy-descent decode — tests Kona's claim directly), and a TRM-style recursive
refiner. Solve-rate is EXACT 81-cell match vs the gold solution (no partial credit).

Data encoding (~/trm_src/data/sudoku-extreme-1k-aug-1000, npy, 81 cells, vocab 11):
  token 0 = pad (unused — all 81 cells present), 1 = blank, 2..10 = digits 1..9.
  input = puzzle (blanks=1 + givens), label = full solution (all 2..10).

GPU: INTERNAL RTX 3090 only — run with
  CUDA_VISIBLE_DEVICES=GPU-7971baff-9583-eaa6-2292-393f930a28f9
(the eGPU GPU-b52387a2 is flaky under sustained load). Conductor must be PAUSED.
"""
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(os.path.expanduser("~/trm_src/data/sudoku-extreme-1k-aug-1000"))
VOCAB = 11          # 0=pad,1=blank,2..10=digits 1..9
SEQ = 81            # 9x9
BLANK = 1
DIGIT0 = 2          # token of digit '1'
N_DIGITS = 9


# --------------------------------------------------------------------------- data
def load_split(split: str, n: int | None, seed: int):
    inp = np.load(DATA / split / "all__inputs.npy", mmap_mode="r")
    lab = np.load(DATA / split / "all__labels.npy", mmap_mode="r")
    N = inp.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)[: (n or N)]
    x = torch.from_numpy(np.asarray(inp[idx], dtype=np.int64))
    y = torch.from_numpy(np.asarray(lab[idx], dtype=np.int64))
    return x, y  # (n,81) tokens in [1..10]; givens where x!=BLANK


def sudoku_violations(grid_digits: torch.Tensor) -> torch.Tensor:
    """grid_digits: (B,81) ints in [0,8] (digit-1). Returns (B,) # of row/col/box
    duplicate violations (0 == a valid Sudoku filling). Used ONLY for post-hoc
    characterisation, never to pick answers (keeps the comparison pure generation)."""
    B = grid_digits.shape[0]
    g = grid_digits.view(B, 9, 9)
    viol = torch.zeros(B, dtype=torch.long, device=grid_digits.device)
    def count(units):  # units: (B, 9groups, 9cells)
        v = torch.zeros(B, dtype=torch.long, device=grid_digits.device)
        for d in range(9):
            cnt = (units == d).sum(-1)        # (B,9groups)
            v = v + (cnt - 1).clamp(min=0).sum(-1)
        return v
    rows = g
    cols = g.transpose(1, 2)
    boxes = g.view(B, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).reshape(B, 9, 9)
    return count(rows) + count(cols) + count(boxes)


# ------------------------------------------------------------------- AR baseline
class ARTransformer(nn.Module):
    """Prefix-LM over [puzzle(81) ; solution(81)]: solution cells attend to the full
    puzzle (bidirectional) + previously-decoded solution cells (causal)."""
    def __init__(self, hidden=256, layers=4, heads=4):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, hidden)
        self.pos = nn.Embedding(2 * SEQ, hidden)
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, batch_first=True,
                                         activation="gelu", dropout=0.0, norm_first=True)
        self.blocks = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(hidden, VOCAB)
        self.register_buffer("mask", self._mask(), persistent=False)

    @staticmethod
    def _mask():
        L = 2 * SEQ
        m = torch.zeros(L, L, dtype=torch.bool)  # True = blocked
        # solution positions (>=SEQ) are causal among the solution segment
        for i in range(SEQ, L):
            for j in range(SEQ, L):
                if j > i:
                    m[i, j] = True
        # puzzle positions (<SEQ) may not see the solution segment
        m[:SEQ, SEQ:] = True
        return m

    def forward(self, x, y):  # x,y: (B,81)
        B = x.shape[0]
        seq = torch.cat([x, y], 1)                    # (B,162)
        h = self.tok(seq) + self.pos(torch.arange(2 * SEQ, device=x.device))[None]
        h = self.blocks(h, mask=self.mask)
        return self.head(h[:, SEQ - 1: 2 * SEQ - 1])  # predict y[t] from pos t-1 → (B,81,V)

    @torch.no_grad()
    def generate(self, x, temperature=0.0, n_samples=1):
        """Returns (B, n_samples, 81) token grids. temperature 0 = greedy."""
        B = x.shape[0]
        x = x.repeat_interleave(n_samples, 0)
        y = torch.full((x.shape[0], SEQ), BLANK, dtype=torch.long, device=x.device)
        for t in range(SEQ):
            logits = self.forward(x, y)[:, t]          # (B*ns, V)
            logits[:, :DIGIT0] = -1e9                   # only digits 2..10 are legal outputs
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1).squeeze(-1)
            y[:, t] = nxt
        return y.view(B, n_samples, SEQ)


# ------------------------------------------------------------------------ eval
def solve_rate(tok_grids: torch.Tensor, gold: torch.Tensor) -> float:
    """tok_grids,(gold): (B,81) tokens. Exact full-grid match."""
    return (tok_grids == gold).all(-1).float().mean().item()


def sc_majority(samples: torch.Tensor) -> torch.Tensor:
    """samples: (B, k, 81) tokens -> (B,81) per-cell majority vote (self-consistency)."""
    B, k, L = samples.shape
    out = torch.zeros(B, L, dtype=torch.long, device=samples.device)
    for c in range(L):
        vals = samples[:, :, c]                       # (B,k)
        oh = F.one_hot(vals, VOCAB).sum(1)            # (B,V)
        out[:, c] = oh.argmax(-1)
    return out


def evaluate_ar(model, xev, yev, device, bs=256, sc_k=32, seed=0):
    model.eval()
    torch.manual_seed(seed)
    greedy_hits, sc_hits, n = 0, 0, xev.shape[0]
    viol_sum, cell_correct, blank_correct, blank_total = 0, 0, 0, 0
    for i in range(0, n, bs):
        xb = xev[i:i + bs].to(device); yb = yev[i:i + bs].to(device)
        g = model.generate(xb, temperature=0.0, n_samples=1)[:, 0]      # greedy
        greedy_hits += (g == yb).all(-1).sum().item()
        viol_sum += sudoku_violations((g - DIGIT0).clamp(0, 8)).sum().item()
        cell_correct += (g == yb).float().sum().item()
        # cell accuracy restricted to BLANK cells (the ones the model must solve)
        blanks = (xb == BLANK)
        blank_correct += ((g == yb) & blanks).float().sum().item()
        blank_total += blanks.float().sum().item()
        s = model.generate(xb, temperature=1.0, n_samples=sc_k)          # SC
        sc = sc_majority(s)
        sc_hits += (sc == yb).all(-1).sum().item()
    return {"ar_greedy_solve_rate": greedy_hits / n,
            "ar_sc{}_solve_rate".format(sc_k): sc_hits / n,
            "ar_greedy_cell_acc": cell_correct / (n * SEQ),
            "ar_greedy_blank_cell_acc": blank_correct / max(1, blank_total),
            "ar_greedy_mean_violations": viol_sum / n,
            "n_eval": n}


# ----------------------------------------------------------------------- train
def train_ar(args, device):
    xtr, ytr = load_split("train", args.train_n, args.seed)
    xev, yev = load_split("test", args.eval_n, args.seed + 1)
    model = ARTransformer(args.hidden, args.layers, args.heads).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps,
                                                pct_start=0.1)
    model.train()
    n = xtr.shape[0]
    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, n, (args.batch,))
        xb = xtr[idx].to(device); yb = ytr[idx].to(device)
        logits = model(xb, yb)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), yb.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"[ar] step={step} loss={loss.item():.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} t={time.time()-t0:.0f}s", flush=True)
    metrics = evaluate_ar(model, xev, yev, device, sc_k=args.sc_k, seed=args.seed)
    metrics.update({"n_params": nparams, "train_seconds": time.time() - t0})
    return model, metrics


# ------------------------------------------------------------- EBT (energy model)
class EBT(nn.Module):
    """Energy E(candidate_solution | puzzle): a bidirectional transformer scoring a
    FULL filled grid against the puzzle. Low energy = constraints satisfied. The
    candidate can be passed as hard tokens OR as soft per-cell digit distributions
    (B,81,9) so we can do continuous-relaxation ENERGY DESCENT (Kona's claim)."""
    def __init__(self, hidden=256, layers=4, heads=4):
        super().__init__()
        self.puz = nn.Embedding(VOCAB, hidden)          # puzzle tokens (0..10)
        self.dig = nn.Embedding(N_DIGITS, hidden)       # candidate digits (0..8)
        self.seg = nn.Embedding(2, hidden)              # 0=puzzle, 1=candidate
        self.pos = nn.Embedding(2 * SEQ, hidden)
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, batch_first=True,
                                         activation="gelu", dropout=0.0, norm_first=True)
        self.blocks = nn.TransformerEncoder(enc, layers)
        self.energy = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def cand_embed(self, cand):
        # cand: (B,81) digit tokens in [0,8]  OR  (B,81,9) soft distribution
        if cand.dim() == 2:
            return self.dig(cand)
        return cand @ self.dig.weight                    # soft: (B,81,9)@(9,H)->(B,81,H)

    def forward(self, x, cand):  # x:(B,81) tokens 0..10 ; cand: see cand_embed
        B = x.shape[0]
        pe = self.puz(x) + self.seg(torch.zeros(SEQ, dtype=torch.long, device=x.device))
        ce = self.cand_embed(cand) + self.seg(torch.ones(SEQ, dtype=torch.long, device=x.device))
        h = torch.cat([pe, ce], 1) + self.pos(torch.arange(2 * SEQ, device=x.device))[None]
        h = self.blocks(h)
        return self.energy(h[:, SEQ:]).mean(1).squeeze(-1)   # (B,) mean per-cell energy


def gold_digits(y):                  # label tokens 2..10 -> digit class 0..8
    return (y - DIGIT0).clamp(0, 8)


def corrupt(yd, xb, frac, rng):
    """Reassign `frac` of the BLANK cells of digit-grid yd to random digits."""
    blanks = (xb == BLANK)
    flip = (torch.rand_like(yd.float()) < frac) & blanks
    rand = torch.randint(0, N_DIGITS, yd.shape, device=yd.device)
    return torch.where(flip, rand, yd)


def train_ebt(args, device):
    xtr, ytr = load_split("train", args.train_n, args.seed)
    xev, yev = load_split("test", args.eval_n, args.seed + 1)
    model = EBT(args.hidden, args.layers, args.heads).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.1)
    model.train(); n = xtr.shape[0]; K = 8; t0 = time.time()
    g = torch.Generator(device=device)
    for step in range(args.steps):
        idx = torch.randint(0, n, (args.batch,))
        xb = xtr[idx].to(device); yd = gold_digits(ytr[idx].to(device))
        e_gold = model(xb, yd)                                 # (B,)
        negs = []
        for _ in range(K):
            frac = 0.1 + 0.5 * torch.rand(1).item()
            negs.append(model(xb, corrupt(yd, xb, frac, g)))
        e_neg = torch.stack(negs, 1)                           # (B,K)
        logits = -torch.cat([e_gold[:, None], e_neg], 1)       # gold should be lowest E
        loss = F.cross_entropy(logits, torch.zeros(xb.shape[0], dtype=torch.long, device=device))
        loss = loss + 1e-3 * (e_gold ** 2).mean()              # tiny anchor vs collapse
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"[ebt] step={step} loss={loss.item():.4f} e_gold={e_gold.mean().item():.3f} "
                  f"e_neg={e_neg.mean().item():.3f} t={time.time()-t0:.0f}s", flush=True)
    metrics = evaluate_ebt(model, xev, yev, device, args)
    metrics.update({"n_params": nparams, "train_seconds": time.time() - t0})
    return model, metrics


@torch.no_grad()
def decode_argmin(model, xb):
    """Greedy coordinate per-cell argmin: for each blank cell pick the digit that
    minimises energy holding others at current best (1 sweep from a uniform start)."""
    B = xb.shape[0]
    yd = gold_digits(xb.clone())                  # givens correct; blanks -> clamp 0
    yd = torch.where(xb == BLANK, torch.zeros_like(yd), yd)
    blanks = (xb == BLANK)
    for c in range(SEQ):
        bmask = blanks[:, c]
        if bmask.sum() == 0:
            continue
        best_e = torch.full((B,), 1e9, device=xb.device); best_d = yd[:, c].clone()
        for d in range(N_DIGITS):
            trial = yd.clone(); trial[:, c] = d
            e = model(xb, trial)
            upd = (e < best_e) & bmask
            best_d = torch.where(upd, torch.full_like(best_d, d), best_d); best_e = torch.where(upd, e, best_e)
        yd[:, c] = torch.where(bmask, best_d, yd[:, c])
    return yd


def decode_descent(model, xb, steps=200, lr=0.3):
    """Continuous-relaxation ENERGY DESCENT (Kona): optimise soft per-cell digit
    logits to minimise E, givens clamped one-hot, then argmax."""
    B = xb.shape[0]
    blanks = (xb == BLANK)
    L = torch.zeros(B, SEQ, N_DIGITS, device=xb.device, requires_grad=True)
    given_oh = F.one_hot(gold_digits(xb).clamp(0, 8), N_DIGITS).float()
    opt = torch.optim.Adam([L], lr=lr)
    for _ in range(steps):
        probs = F.softmax(L, -1)
        soft = torch.where(blanks[..., None], probs, given_oh)   # clamp givens
        e = model(xb, soft).mean()
        opt.zero_grad(); e.backward(); opt.step()
    with torch.no_grad():
        probs = F.softmax(L, -1)
        out = torch.where(blanks, probs.argmax(-1), gold_digits(xb).clamp(0, 8))
    return out


@torch.no_grad()
def evaluate_ebt(model, xev, yev, device, args, bs=128):
    model.eval()
    res = {decoder: {"hits": 0, "blank_correct": 0.0, "viol": 0}
           for decoder in ["argmin", "descent"]}
    n = xev.shape[0]; blank_total = 0.0
    for i in range(0, n, bs):
        xb = xev[i:i + bs].to(device); yb = yev[i:i + bs].to(device); yd = gold_digits(yb)
        blanks = (xb == BLANK); blank_total += blanks.float().sum().item()
        am = decode_argmin(model, xb)
        with torch.enable_grad():
            dc = decode_descent(model, xb, steps=args.descent_steps, lr=args.descent_lr)
        for name, pred in [("argmin", am), ("descent", dc)]:
            res[name]["hits"] += (pred == yd).all(-1).sum().item()
            res[name]["blank_correct"] += ((pred == yd) & blanks).float().sum().item()
            res[name]["viol"] += sudoku_violations(pred).sum().item()
    out = {"n_eval": n}
    for name in res:
        out[f"ebt_{name}_solve_rate"] = res[name]["hits"] / n
        out[f"ebt_{name}_blank_cell_acc"] = res[name]["blank_correct"] / max(1, blank_total)
        out[f"ebt_{name}_mean_violations"] = res[name]["viol"] / n
    out["ebt_best_solve_rate"] = max(out["ebt_argmin_solve_rate"], out["ebt_descent_solve_rate"])
    return out


# ------------------------------------------------------- TRM-style recursive refiner
class Refiner(nn.Module):
    """Tiny recursive refiner (TRM recipe): hold current-answer y + latent z; iterate
    z <- block(x,y,z) then y <- head(z); output final y. Non-AR, holistic."""
    def __init__(self, hidden=256, layers=2, heads=4, n_cycles=8):
        super().__init__()
        self.n_cycles = n_cycles
        self.puz = nn.Embedding(VOCAB, hidden)
        self.dig = nn.Embedding(N_DIGITS, hidden)
        self.pos = nn.Embedding(SEQ, hidden)
        self.zin = nn.Linear(hidden * 3, hidden)
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, batch_first=True,
                                         activation="gelu", dropout=0.0, norm_first=True)
        self.block = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(hidden, N_DIGITS)

    def forward(self, x, n_cycles=None, deep_supervision=False):
        B = x.shape[0]; dev = x.device
        T = n_cycles or self.n_cycles
        xe = self.puz(x) + self.pos(torch.arange(SEQ, device=dev))[None]
        # init y from the puzzle givens (blanks -> digit 0), z from zeros
        yd = torch.where(x == BLANK, torch.zeros_like(x), gold_digits(x).clamp(0, 8))
        z = torch.zeros(B, SEQ, xe.shape[-1], device=dev)
        outs = []
        for t in range(T):
            ye = self.dig(yd)
            z = self.block(self.zin(torch.cat([xe, ye, z], -1)))
            logits = self.head(z)
            yd = logits.argmax(-1).detach()                  # update current answer
            if deep_supervision or t == T - 1:
                outs.append(logits)
        return outs if deep_supervision else outs[-1]


def train_refiner(args, device):
    xtr, ytr = load_split("train", args.train_n, args.seed)
    xev, yev = load_split("test", args.eval_n, args.seed + 1)
    model = Refiner(args.hidden, max(2, args.layers // 2), args.heads, args.n_cycles).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.1)
    model.train(); n = xtr.shape[0]; t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, n, (args.batch,))
        xb = xtr[idx].to(device); yd = gold_digits(ytr[idx].to(device))
        outs = model(xb, deep_supervision=True)              # list of (B,81,9)
        loss = sum(F.cross_entropy(o.reshape(-1, N_DIGITS), yd.reshape(-1)) for o in outs) / len(outs)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"[refiner] step={step} loss={loss.item():.4f} t={time.time()-t0:.0f}s", flush=True)
    metrics = evaluate_refiner(model, xev, yev, device)
    metrics.update({"n_params": nparams, "train_seconds": time.time() - t0})
    return model, metrics


@torch.no_grad()
def evaluate_refiner(model, xev, yev, device, bs=256):
    model.eval(); n = xev.shape[0]; hits = 0; blank_correct = 0.0; blank_total = 0.0; viol = 0
    for i in range(0, n, bs):
        xb = xev[i:i + bs].to(device); yd = gold_digits(yev[i:i + bs].to(device))
        pred = model(xb).argmax(-1)
        blanks = (xb == BLANK); blank_total += blanks.float().sum().item()
        hits += (pred == yd).all(-1).sum().item()
        blank_correct += ((pred == yd) & blanks).float().sum().item()
        viol += sudoku_violations(pred).sum().item()
    return {"refiner_solve_rate": hits / n,
            "refiner_blank_cell_acc": blank_correct / max(1, blank_total),
            "refiner_mean_violations": viol / n, "n_eval": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ar", "ebt", "refiner"], default="ar")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train_n", type=int, default=200000)
    ap.add_argument("--eval_n", type=int, default=1000)
    ap.add_argument("--sc_k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--descent_steps", type=int, default=200)  # EBT energy-descent iters
    ap.add_argument("--descent_lr", type=float, default=0.3)
    ap.add_argument("--n_cycles", type=int, default=8)         # refiner recursion depth
    ap.add_argument("--out", default="results/sudoku_energy_vs_ar_v1.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} cuda_devices={torch.cuda.device_count()} model={args.model}", flush=True)
    assert device == "cuda", "refusing to run on CPU — pin the internal 3090"

    if args.model == "ar":
        _, metrics = train_ar(args, device)
    elif args.model == "ebt":
        _, metrics = train_ebt(args, device)
    elif args.model == "refiner":
        _, metrics = train_refiner(args, device)

    metrics.update({"model": args.model, "args": vars(args),
                    "gpu": torch.cuda.get_device_name(0)})
    # headroom positive control verdict (AR only)
    if args.model == "ar":
        sc = metrics[f"ar_sc{args.sc_k}_solve_rate"]; gr = metrics["ar_greedy_solve_rate"]
        metrics["headroom_gate_pass"] = bool(sc < 0.75)
        metrics["headroom_note"] = (
            f"AR greedy={gr:.3f} SC{args.sc_k}={sc:.3f}; "
            + ("PASS — weak-AR regime, real headroom (gate <0.75)"
               if sc < 0.75 else
               "FAIL — AR+SC too strong (>0.75), regime polluted, ABORT"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(args.out, "w"), indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
