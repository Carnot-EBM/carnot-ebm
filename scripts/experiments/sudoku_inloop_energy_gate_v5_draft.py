"""DRAFT (#3, operator-facing): energy as an IN-LOOP feasibility gate on Sudoku.

THE UNTESTED THIRD MODE
-----------------------
v1->v4 tested energy two ways and BOTH lost as generators:
  - energy DESCENT (rand-init and refiner-init)  -> ~0 / no lift over the refiner
  - energy RERANK of K finished refiner samples   -> HURT the refiner (0.18->0.12)
What we NEVER tested is the third mode the operator named: energy as a per-step
FEASIBILITY GATE *inside* the decode loop -- not minimising energy, not scoring
finished candidates, but using the perfect scorer to PRUNE the refiner's next-cell
choice to the feasible set. This is verifier-guided generation: the amortized map
proposes, the energy filters, per step.

Note v1's sudoku_violations docstring deliberately kept the energy "post-hoc,
never to pick answers, to keep the comparison pure generation." This draft
INTENTIONALLY crosses that line -- because the whole question is whether the
perfect scorer adds *generative* value when used in-loop. We are no longer testing
"pure generation"; we are testing the Carnot hybrid (amortized generator + energy
as the in-loop verifier).

PROCEDURES COMPARED (same trained Refiner):
  (A) refiner_greedy            -- argmax per cell, feasibility ignored (v4 mode 2)
  (B) refiner_energy_gated      -- NEW mode 5: decode blanks in refiner-confidence
                                   order; at each cell pick the highest-prob digit
                                   that does NOT conflict (row/col/box) with already
                                   -committed cells; fall back to top-1 if none.
  (C) energy_rerank@K (optional)-- v4 mode 4, for reference (known <= greedy).

FALSIFICATION GATE: mode (B) > mode (A) by a clear margin (>= +0.03 solve-rate,
non-overlapping across >=3 seeds) => energy adds GENERATIVE value as an in-loop
gate where descent and rerank failed. If (B) ~ (A), the in-loop use of the perfect
scorer does not help either, and energy-as-anything-but-a-post-hoc-scorer is dead
on this task.

The feasibility oracle here is the TRUE constraint function (the perfect scorer we
established energy IS on Sudoku). A follow-up should swap in the LEARNED EBT energy
per-step to test whether a learned verifier can substitute for the exact one.

DRAFT. Run in a paused-conductor internal-GPU window:
  CUDA_VISIBLE_DEVICES=GPU-7971baff-... .venv/bin/python \
    scripts/experiments/sudoku_inloop_energy_gate_v5_draft.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sudoku_energy_vs_ar_v1 import (  # noqa: E402
    BLANK,
    DIGIT0,
    Refiner,
    gold_digits,
    load_split,
    sudoku_violations,
)
from sudoku_amortized_capstone_v4 import train_refiner_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "results" / "sudoku_inloop_energy_gate_v5.json"

# Precompute the 20 row/col/box peers of each of the 81 cells (constant geometry).
_ROW = [c // 9 for c in range(81)]
_COL = [c % 9 for c in range(81)]
_BOX = [(_ROW[c] // 3) * 3 + (_COL[c] // 3) for c in range(81)]
PEERS: list[list[int]] = [
    [d for d in range(81) if d != c and (_ROW[d] == _ROW[c] or _COL[d] == _COL[c] or _BOX[d] == _BOX[c])]
    for c in range(81)
]


@torch.no_grad()
def refiner_greedy(refiner: torch.nn.Module, xb: torch.Tensor) -> torch.Tensor:
    """(B,81) digit classes 0..8, argmax per cell, feasibility ignored."""

    return refiner(xb).argmax(-1)


@torch.no_grad()
def refiner_energy_gated(refiner: torch.nn.Module, xb: torch.Tensor) -> torch.Tensor:
    """Mode 5: confidence-ordered, feasibility-gated decode of the blank cells."""

    probs = F.softmax(refiner(xb), -1)  # (B,81,9)
    B = xb.shape[0]
    blanks = xb == BLANK  # (B,81)
    given = ~blanks
    grid = torch.full((B, 81), -1, dtype=torch.long, device=xb.device)
    grid[given] = xb[given] - DIGIT0  # token (>=2) -> digit class 0..8
    conf = probs.max(-1).values.masked_fill(given, -1.0)  # decode blanks only
    order = conf.argsort(dim=1, descending=True)  # (B,81) per-sample cell order
    ranked = probs.argsort(dim=-1, descending=True)  # (B,81,9) per-cell digit order
    grid_cpu = grid.cpu().tolist()
    order_cpu = order.cpu().tolist()
    ranked_cpu = ranked.cpu().tolist()
    for b in range(B):
        g = grid_cpu[b]
        for c in order_cpu[b]:
            if g[c] >= 0:  # given or already filled
                continue
            used = {g[p] for p in PEERS[c] if g[p] >= 0}
            chosen = next((d for d in ranked_cpu[b][c] if d not in used), ranked_cpu[b][c][0])
            g[c] = chosen
    return torch.tensor(grid_cpu, dtype=torch.long, device=xb.device)


@torch.no_grad()
def _solve_rate(pred: torch.Tensor, yd: torch.Tensor) -> float:
    return (pred == yd).all(-1).float().mean().item()


def _evaluate(refiner: torch.nn.Module, xev: torch.Tensor, yev: torch.Tensor, *, bs: int) -> dict:
    refiner.eval()
    yd_all = gold_digits(yev.clone())
    n = xev.shape[0]
    greedy_hits = gated_hits = 0
    gated_viol_total = 0
    for i in range(0, n, bs):
        xb = xev[i : i + bs]
        yd = yd_all[i : i + bs]
        g_greedy = refiner_greedy(refiner, xb)
        g_gated = refiner_energy_gated(refiner, xb)
        greedy_hits += (g_greedy == yd).all(-1).sum().item()
        gated_hits += (g_gated == yd).all(-1).sum().item()
        gated_viol_total += int(sudoku_violations(g_gated).sum().item())
    return {
        "refiner_greedy_solve_rate": greedy_hits / n,
        "refiner_energy_gated_solve_rate": gated_hits / n,
        "gated_mean_violations": gated_viol_total / n,
        "n_eval": n,
    }


def _args(seed: int) -> SimpleNamespace:
    # Arg surface matching v4.train_refiner_model: hidden/layers/heads/n_cycles/
    # lr/r_steps/batch/log_every.
    return SimpleNamespace(
        hidden=256, layers=4, heads=4, n_cycles=8, r_steps=4000, lr=3e-4,
        batch=128, log_every=500, seed=seed,
    )


def run(seeds: list[int], *, n_train: int, n_eval: int, bs: int, write: bool = True) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        artifact = {
            "experiment": "sudoku_inloop_energy_gate_v5_draft",
            "honest_verdict": "blocked_no_cuda",
            "inference_substrate": "none_blocked_preflight",
            "preconditions_checked": [{"resource": "cuda_available", "available": False}],
            "duration_s": 0.0,
        }
        if write:
            OUTPUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")
        return artifact

    started = time.time()
    per_seed = []
    for seed in seeds:
        torch.manual_seed(seed)
        xtr, ytr = load_split("train", n_train, seed)
        xev, yev = load_split("test", n_eval, seed + 1000)
        xtr, ytr, xev, yev = (t.to(device) for t in (xtr, ytr, xev, yev))
        a = _args(seed)
        refiner = train_refiner_model(xtr, ytr, a, device)
        res = _evaluate(refiner, xev, yev, bs=bs)
        res["seed"] = seed
        res["delta_gated_minus_greedy"] = (
            res["refiner_energy_gated_solve_rate"] - res["refiner_greedy_solve_rate"]
        )
        per_seed.append(res)
        print(f"[seed {seed}] greedy={res['refiner_greedy_solve_rate']:.4f} "
              f"gated={res['refiner_energy_gated_solve_rate']:.4f} "
              f"delta={res['delta_gated_minus_greedy']:+.4f}", flush=True)

    deltas = [r["delta_gated_minus_greedy"] for r in per_seed]
    mean_delta = sum(deltas) / len(deltas)
    all_positive = all(d >= 0.03 for d in deltas)
    verdict = (
        f"complete: inloop_energy_gate_HELPS_meandelta{mean_delta:+.4f}_allseeds_ge_0.03"
        if all_positive
        else f"complete: inloop_energy_gate_no_clear_lift_meandelta{mean_delta:+.4f}"
    )
    artifact = {
        "experiment": "sudoku_inloop_energy_gate_v5_draft",
        "title": "sudoku_inloop_energy_feasibility_gate",
        "honest_verdict": verdict,
        "inference_substrate": "live_gpu_training_plus_true_energy_inloop_gate",
        "mode": "energy as per-step feasibility gate (NOT descent, NOT rerank)",
        "feasibility_oracle": "true sudoku_violations (the perfect scorer)",
        "mean_delta_gated_minus_greedy": mean_delta,
        "per_seed": per_seed,
        "seeds": seeds,
        "falsification_gate": "gated > greedy by >= +0.03 on every seed",
        "preconditions_checked": [{"resource": "cuda_available", "available": True}],
        "duration_s": time.time() - started,
        "caveat": (
            "DRAFT. Uses the TRUE constraint function as the in-loop oracle (the perfect "
            "scorer energy IS on Sudoku). Follow-up: swap the LEARNED EBT energy per-step. "
            "Tiny from-scratch matched-compute -- the ORDERING (gated vs greedy) is the "
            "result, not absolute solve rates."
        ),
    }
    if write:
        OUTPUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")
    return artifact


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_eval", type=int, default=512)
    ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()
    art = run(args.seeds, n_train=args.n_train, n_eval=args.n_eval, bs=args.bs)
    print(f"-> {art['honest_verdict']}")
    return 0 if str(art["honest_verdict"]).startswith("complete:") else 1


if __name__ == "__main__":
    raise SystemExit(main())
