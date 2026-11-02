"""DRAFT (#4 + #3-retest, operator-facing): energy as a TEACHER, on a FAIR base.

WHY CONSOLIDATED
----------------
#3 (energy as an in-loop feasibility gate) came back +0.0007 -- but the refiner
was grossly undertrained (~3% solve; r_steps=4000/train_n=10000) vs v4's ~18%
(r_steps=15000/train_n=400000).  At a 3% floor the gate test is underpowered.
Both #3-retest and #4 need the SAME well-trained base refiner, so this trains it
ONCE (v4 config) and compares, on that one base:

  (A) base_greedy              -- the real ~18% amortized baseline
  (B) base_energy_gated        -- #3 RETEST: per-step feasibility gate, fair base
  (C) energy_distilled_greedy  -- #4: STaR-style self-distillation where the
                                  ENERGY (sudoku_violations), NOT gold labels,
                                  selects the target. Generate K samples/puzzle,
                                  keep the lowest-violation one, train the refiner
                                  to imitate it.  NO ground-truth answers used.
  (D) gold_distilled_greedy    -- UPPER BOUND: same self-distillation but selecting
                                  the gold answer. Bounds how much of the label
                                  ceiling the energy-only signal (C) recovers.

THE #4 HYPOTHESIS (the constructive flip of the energy-as-generator negatives).
Energy-DESCENT, energy-RERANK, and the energy-GATE all failed to add generative
value at INFERENCE time.  #4 asks the dual question: does energy add value at
TRAINING time -- can the perfect scorer, with no labels, TEACH the amortized map
to generate better?  This is where amortized inference is supposed to win, so it
is the most promising remaining use of the energy.

FALSIFICATION GATES:
  #4 works  := (C) > (A) by >= +0.03 (energy-only self-distillation lifts solve).
  #4 strong := (C) recovers >= 50% of the (D)-(A) gold-ceiling gap.
  #3 retest := (B) > (A) by >= +0.03 on a FAIR base (else the gate is confirmed dead).

DRAFT.  Run in a paused-conductor internal-GPU window.  Defaults to 1 seed + full
v4 config (~25-40 min); add --seeds for replication, --fast for a pipeline smoke.
  CUDA_VISIBLE_DEVICES=GPU-7971baff-... .venv/bin/python \
    scripts/experiments/sudoku_energy_teacher_v6_draft.py --seeds 0
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sudoku_energy_vs_ar_v1 import (  # noqa: E402
    SEQ,
    Refiner,
    gold_digits,
    load_split,
    sudoku_violations,
)
from sudoku_amortized_capstone_v4 import (  # noqa: E402
    refiner_greedy,
    refiner_samples,
    train_refiner_model,
)
from sudoku_inloop_energy_gate_v5_draft import refiner_energy_gated  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "results" / "sudoku_energy_teacher_v6.json"
N_DIGITS = 9


def _base_args(seed: int, *, fast: bool) -> SimpleNamespace:
    # v4's real config (r_steps=15000, train_n=400000, batch=256) -> ~18% base.
    return SimpleNamespace(
        hidden=256, layers=4, heads=4, n_cycles=8, lr=3e-4,
        r_steps=1500 if fast else 15000,
        batch=128 if fast else 256,
        log_every=500, seed=seed,
    )


@torch.no_grad()
def _eval_greedy(refiner: torch.nn.Module, xev: torch.Tensor, yd: torch.Tensor, *, bs: int) -> float:
    refiner.eval()
    hits = n = 0
    for i in range(0, xev.shape[0], bs):
        g = refiner_greedy(refiner, xev[i : i + bs])
        hits += (g == yd[i : i + bs]).all(-1).sum().item()
        n += g.shape[0]
    return hits / n


@torch.no_grad()
def _eval_gated(refiner: torch.nn.Module, xev: torch.Tensor, yd: torch.Tensor, *, bs: int) -> float:
    refiner.eval()
    hits = n = 0
    for i in range(0, xev.shape[0], bs):
        g = refiner_energy_gated(refiner, xev[i : i + bs])
        hits += (g == yd[i : i + bs]).all(-1).sum().item()
        n += g.shape[0]
    return hits / n


def _self_distill(
    refiner: torch.nn.Module,
    xtr: torch.Tensor,
    *,
    selector: str,  # "energy" (no labels) or "gold" (upper bound)
    ytr_digits: torch.Tensor | None,
    device: str,
    steps: int,
    K: int,
    temp: float,
    lr: float,
    batch: int,
) -> torch.nn.Module:
    """STaR-style self-distillation; target chosen by ENERGY or GOLD."""

    # v2 fixes (the v1 collapse was a HARNESS bug -- the gold control also fell):
    #  1. DEEP-SUPERVISION-preserving loss (sum CE over the recurrent cycle outputs,
    #     exactly how train_refiner_model trains the refiner). v1 used a single
    #     forward, which degraded the recurrent model regardless of selector.
    #  2. ANTI-FORGETTING ANCHOR (no gold): for non-certified puzzles, the target is
    #     the FROZEN BASE's own greedy output, so the model is held to what it already
    #     knew on the ~95% it cannot yet certify, and only MOVES on certified puzzles.
    #  3. Higher K raises the certified yield. None of this leaks gold into the energy
    #     arm -- the anchor is the base model, not the labels.
    model = copy.deepcopy(refiner)
    frozen = refiner  # the base BEFORE distillation; the anti-forgetting anchor
    frozen.eval()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    n = xtr.shape[0]
    t0 = time.time()
    yields: list[float] = []
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        xb = xtr[idx]
        B = xb.shape[0]
        if selector == "gold":
            # UPPER BOUND: deep-supervised SFT on the true solution (every puzzle).
            target = ytr_digits[idx]  # (B,81)
            yields.append(1.0)
        else:
            # ENERGY-AS-TEACHER (RFT): certified-correct self-samples where available,
            # frozen-base anchor elsewhere. NO gold.
            with torch.no_grad():
                samples = refiner_samples(model, xb, K, temp)  # (B,K,81) digit classes
                viol = sudoku_violations(samples.reshape(B * K, SEQ)).reshape(B, K)
                min_viol, best = viol.min(1)  # (B,), (B,)
                cert = samples[torch.arange(B, device=device), best]  # (B,81) best sample
                certified = min_viol == 0  # verifier-CERTIFIED correct (unique solution)
                anchor = frozen(xb).argmax(-1)  # (B,81) base behaviour (anti-forgetting)
            target = torch.where(certified[:, None], cert, anchor)
            yields.append(float(certified.float().mean().item()))
        outs = model(xb, deep_supervision=True)  # list of (B,81,9), one per cycle
        loss = sum(
            F.cross_entropy(o.reshape(-1, N_DIGITS), target.reshape(-1)) for o in outs
        ) / len(outs)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == steps - 1:
            print(f"[distill:{selector}] step={step} loss={loss.item():.4f} "
                  f"certified_yield={yields[-1]:.3f} t={time.time()-t0:.0f}s", flush=True)
    model._distill_mean_yield = sum(yields) / len(yields) if yields else 0.0  # type: ignore[attr-defined]
    return model


def run(seeds: list[int], *, fast: bool, n_eval: int, bs: int, K: int, temp: float,
        distill_steps: int, distill_lr: float, base_steps: int | None = None,
        output_path: Path | None = None, write: bool = True) -> dict:
    out_path = output_path or (OUTPUT_PATH if base_steps is None else OUTPUT_PATH.with_name(
        OUTPUT_PATH.stem + f"_base{base_steps}.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        art = {"experiment": "sudoku_energy_teacher_v6_draft", "honest_verdict": "blocked_no_cuda",
               "inference_substrate": "none_blocked_preflight", "duration_s": 0.0}
        if write:
            out_path.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
        return art

    started = time.time()
    per_seed = []
    for seed in seeds:
        torch.manual_seed(seed)
        a = _base_args(seed, fast=fast)
        if base_steps is not None:
            a.r_steps = base_steps  # #4 v3: deliberately UNDER-train for headroom
        xtr, ytr = load_split("train", 20000 if fast else 400000, seed)
        xev, yev = load_split("test", n_eval, seed + 1000)
        xtr, ytr, xev, yev = (t.to(device) for t in (xtr, ytr, xev, yev))
        yd_ev = gold_digits(yev.clone())
        ytr_digits = gold_digits(ytr.clone())

        base = train_refiner_model(xtr, ytr, a, device)
        base_greedy = _eval_greedy(base, xev, yd_ev, bs=bs)
        base_gated = _eval_gated(base, xev, yd_ev, bs=bs)
        energy_model = _self_distill(base, xtr, selector="energy", ytr_digits=None, device=device,
                                     steps=distill_steps, K=K, temp=temp, lr=distill_lr, batch=a.batch)
        energy_greedy = _eval_greedy(energy_model, xev, yd_ev, bs=bs)
        gold_model = _self_distill(base, xtr, selector="gold", ytr_digits=ytr_digits, device=device,
                                   steps=distill_steps, K=K, temp=temp, lr=distill_lr, batch=a.batch)
        gold_greedy = _eval_greedy(gold_model, xev, yd_ev, bs=bs)

        rec = {
            "seed": seed,
            "base_greedy": base_greedy,
            "base_energy_gated": base_gated,
            "energy_distilled_greedy": energy_greedy,
            "gold_distilled_greedy": gold_greedy,
            "energy_certified_yield": getattr(energy_model, "_distill_mean_yield", None),
            "delta_energy_teacher": energy_greedy - base_greedy,
            "delta_gate_retest": base_gated - base_greedy,
            "gold_ceiling_gap": gold_greedy - base_greedy,
            "frac_of_gold_ceiling_recovered": (
                (energy_greedy - base_greedy) / (gold_greedy - base_greedy)
                if gold_greedy - base_greedy > 1e-6 else None
            ),
        }
        per_seed.append(rec)
        print(f"[seed {seed}] base={base_greedy:.4f} gated={base_gated:.4f} "
              f"energy_taught={energy_greedy:.4f} gold_taught={gold_greedy:.4f} "
              f"d_teacher={rec['delta_energy_teacher']:+.4f}", flush=True)

    def _mean(k: str) -> float:
        vals = [r[k] for r in per_seed if r[k] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    md_teacher = _mean("delta_energy_teacher")
    md_gate = _mean("delta_gate_retest")
    teacher_works = all(r["delta_energy_teacher"] >= 0.03 for r in per_seed)
    gate_works = all(r["delta_gate_retest"] >= 0.03 for r in per_seed)
    verdict = (
        f"complete: energy_teacher_{'WORKS' if teacher_works else 'no_lift'}"
        f"_dteacher{md_teacher:+.4f}_gate_{'works' if gate_works else 'dead'}{md_gate:+.4f}"
        f"_goldfrac{_mean('frac_of_gold_ceiling_recovered'):.2f}"
    )
    art = {
        "experiment": "sudoku_energy_teacher_v6_draft",
        "title": "sudoku_energy_as_teacher_plus_gate_retest",
        "honest_verdict": verdict,
        "inference_substrate": "live_gpu_training_energy_self_distillation_plus_gate_retest",
        "mean_delta_energy_teacher": md_teacher,
        "mean_delta_gate_retest": md_gate,
        "mean_frac_gold_ceiling_recovered": _mean("frac_of_gold_ceiling_recovered"),
        "per_seed": per_seed,
        "seeds": seeds,
        "fast_mode": fast,
        "config": {"K": K, "temp": temp, "distill_steps": distill_steps, "distill_lr": distill_lr},
        "falsification_gates": {
            "energy_teacher_works": "energy_distilled > base by >= +0.03 every seed",
            "energy_teacher_strong": ">= 50% of the gold-ceiling gap recovered (no labels)",
            "gate_retest": "base_energy_gated > base by >= +0.03 (else the in-loop gate is dead)",
        },
        "preconditions_checked": [{"resource": "cuda_available", "available": True}],
        "duration_s": time.time() - started,
        "caveat": (
            "DRAFT. Energy selector uses the TRUE sudoku_violations (the perfect scorer); a "
            "follow-up should use the LEARNED EBT energy to test a learned teacher. Tiny "
            "matched-compute: the ORDERING (energy-taught vs base, and vs gold upper bound) is "
            "the result, not absolute solve rates. fast_mode shrinks training for pipeline smoke."
        ),
    }
    if write:
        out_path.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    return art


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--fast", action="store_true", help="shrink training for a pipeline smoke")
    ap.add_argument("--n_eval", type=int, default=512)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--distill_steps", type=int, default=2000)
    ap.add_argument("--distill_lr", type=float, default=1e-4)
    ap.add_argument("--base_steps", type=int, default=None,
                    help="#4 v3: under-train the base to this many steps for headroom")
    ap.add_argument("--output-path", type=Path, default=None)
    args = ap.parse_args()
    art = run(args.seeds, fast=args.fast, n_eval=args.n_eval, bs=args.bs, K=args.K, temp=args.temp,
              distill_steps=args.distill_steps, distill_lr=args.distill_lr, base_steps=args.base_steps,
              output_path=args.output_path)
    print(f"-> {art['honest_verdict']}")
    return 0 if str(art["honest_verdict"]).startswith("complete:") else 1


if __name__ == "__main__":
    raise SystemExit(main())
