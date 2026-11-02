#!/usr/bin/env python3
"""Verifier-graft v3 Stage A — HEADROOM PROBE (the positive control).

Question: on the FULL Sudoku-Extreme test set (4227 puzzles), with diverse sampling,
is there real recoverable headroom for a reranker/verifier — i.e. oracle@K >> vote@1?

This is the cheap gate that decides whether the expensive native-RFT Stage B is worth
building. v2's null was invalid because it was run on _valsmall (64 puzzles) with
near-deterministic sampling, so oracle@K - vote@1 = 1 puzzle (zero power). Here we use the
full test set and a real diversity source.

Diversity sources (no Sudoku decoding needed — the gate is token-level exact-match):
  1. The TRM's per-supervision-step argmax predictions (the recursion "thinking trajectory"
     gives distinct candidates).
  2. K temperature-sampled candidates from the final-step logits, over a temperature sweep.

Metrics per puzzle (exact-match vs label on the non-ignore positions):
  greedy@1  = final-step argmax correct
  vote@1    = majority candidate (mode of the pool) correct  [self-consistency baseline]
  oracle@K  = ANY candidate in the pool correct              [the ceiling a selector could reach]
  headroom  = oracle@K - vote@1                              [the gate; need >= 0.05]

Outputs results/trm_runs/v3_headroom_probe.json + logs to v3_probe.log.
GATE: if max headroom across the temperature sweep >= 0.05 -> Stage B greenlit; else the
honest finding is "no recoverable rerank headroom on this baseline" (move to a harder domain).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import torch

# The TRM checkpoint embeds a pathlib.PosixPath, which PyTorch 2.6's weights_only=True
# default refuses to unpickle. We created + trust this checkpoint, so force weights_only=False
# for this probe's loads (same effective behaviour as Lightning's resume path that wrote it).
_orig_torch_load = torch.load


def _trusting_load(*a, **k):
    k["weights_only"] = False  # force (Lightning passes weights_only=True explicitly)
    return _orig_torch_load(*a, **k)


torch.load = _trusting_load

NANO_TRM = "/home/ianblenke/github.com/ianblenke/carnot/nano-trm"
sys.path.insert(0, NANO_TRM)

from src.nn.sudoku_evaluator import SudokuEvaluator  # noqa: E402
from src.nn.utils.constants import IGNORE_LABEL_ID  # noqa: E402

STABLE = "/home/ianblenke/github.com/ianblenke/carnot/results/trm_runs/sudoku_extreme_baseline/last.ckpt"
DATA_DIR = f"{NANO_TRM}/data/sudoku_extreme_1k_aug_1k"
OUT = "/home/ianblenke/github.com/ianblenke/carnot/results/trm_runs/v3_headroom_probe.json"


def exact_correct(cand: torch.Tensor, label: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """cand,label,mask: [B,S]. Return [B] bool: all non-ignore positions match."""
    ok = (cand == label) | (~mask)
    return ok.all(dim=-1)


@torch.no_grad()
def run_batch(model, batch, device, K, temps):
    """One forward (carry loop) per batch; sample candidates at each temperature.

    Returns per-temperature dict T -> dict(greedy, vote, oracle) of [B] bool tensors,
    plus n (batch size). step-argmax candidates are shared across temperatures.
    """
    batch = {k: v.to(device) for k, v in batch.items()}
    carry = model.initial_carry(batch)
    step_preds = []
    final_logits = None
    steps = 0
    while True:
        carry, outputs = model.forward(carry, batch)
        logits = outputs["logits"]  # [B,S,V]
        step_preds.append(logits.argmax(dim=-1))  # [B,S]
        final_logits = logits
        steps += 1
        if carry.halted.all() or steps > 64:
            break

    label = carry.current_data["output"]  # [B,S]
    mask = label != IGNORE_LABEL_ID
    B, S = label.shape

    greedy = step_preds[-1]  # final-step argmax
    greedy_ok = exact_correct(greedy, label, mask)  # [B]

    # de-dup the step-trajectory candidates (shared across temps)
    step_stack = torch.stack(step_preds, dim=0)  # [n_step, B, S]

    out = {}
    probs_base = torch.softmax(final_logits, dim=-1)  # cache once; re-temp below
    for T in temps:
        # K temperature samples from final logits
        if T == 1.0:
            probs = probs_base
        else:
            probs = torch.softmax(final_logits / T, dim=-1)
        samples = []
        flat = probs.reshape(-1, probs.shape[-1])  # [B*S, V]
        for _ in range(K):
            s = torch.multinomial(flat, 1).reshape(B, S)  # [B,S]
            samples.append(s)
        pool = torch.cat([step_stack, torch.stack(samples, dim=0)], dim=0)  # [n_cand,B,S]
        n_cand = pool.shape[0]

        # per-candidate correctness [n_cand, B]
        corr = torch.stack(
            [exact_correct(pool[c], label, mask) for c in range(n_cand)], dim=0
        )
        oracle_ok = corr.any(dim=0)  # [B]

        # vote: mode candidate per puzzle (CPU, small pool)
        pool_cpu = pool.cpu()
        corr_cpu = corr.cpu()
        vote_ok = torch.zeros(B, dtype=torch.bool)
        for b in range(B):
            keys = [tuple(pool_cpu[c, b].tolist()) for c in range(n_cand)]
            cnt = Counter(keys)
            mode_key, _ = cnt.most_common(1)[0]
            # is the mode vector correct? find a candidate index with that key
            for c in range(n_cand):
                if keys[c] == mode_key:
                    vote_ok[b] = bool(corr_cpu[c, b])
                    break
        out[T] = {
            "greedy": greedy_ok.cpu(),
            "vote": vote_ok,
            "oracle": oracle_ok.cpu(),
            "n_cand": n_cand,
        }
    return out, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=STABLE)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--temps", type=str, default="0.7,1.0,1.3")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit-batches", type=int, default=0, help="0 = full test set (smoke: 2)")
    ap.add_argument("--out", type=str, default=OUT)
    args = ap.parse_args()
    temps = [float(t) for t in args.temps.split(",")]

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device} K={args.K} temps={temps} bs={args.batch_size}", flush=True)

    ev = SudokuEvaluator(
        checkpoint_path=args.checkpoint, data_dir=DATA_DIR, batch_size=args.batch_size,
        device="auto", eval_split="test",
    )
    ev.datamodule.setup("test")
    loader = ev.datamodule.test_dataloader()
    model = ev.model
    model.eval()

    acc = {T: {"greedy": 0, "vote": 0, "oracle": 0, "n": 0, "n_cand": 0} for T in temps}
    nb = 0
    for bi, batch in enumerate(loader):
        if args.limit_batches and bi >= args.limit_batches:
            break
        out, B = run_batch(model, batch, ev.device, args.K, temps)
        for T in temps:
            acc[T]["greedy"] += int(out[T]["greedy"].sum())
            acc[T]["vote"] += int(out[T]["vote"].sum())
            acc[T]["oracle"] += int(out[T]["oracle"].sum())
            acc[T]["n"] += B
            acc[T]["n_cand"] = out[T]["n_cand"]
        nb += 1
        if bi % 5 == 0:
            n = acc[temps[0]]["n"]
            print(f"[probe] batch {bi} n={n} "
                  + " ".join(
                      f"T{T}:o={acc[T]['oracle']/max(1,acc[T]['n']):.3f}/v={acc[T]['vote']/max(1,acc[T]['n']):.3f}"
                      for T in temps),
                  flush=True)

    results = {"K": args.K, "temps": temps, "batch_size": args.batch_size,
               "n_batches": nb, "duration_s": round(time.time() - t0, 1),
               "checkpoint": args.checkpoint, "per_temperature": {}}
    best_headroom = -1.0
    for T in temps:
        n = max(1, acc[T]["n"])
        g, v, o = acc[T]["greedy"]/n, acc[T]["vote"]/n, acc[T]["oracle"]/n
        hr = o - v
        best_headroom = max(best_headroom, hr)
        results["per_temperature"][str(T)] = {
            "n": acc[T]["n"], "n_candidates_per_puzzle": acc[T]["n_cand"],
            "greedy_at_1": round(g, 4), "vote_at_1": round(v, 4),
            "oracle_at_k": round(o, 4), "headroom_oracle_minus_vote": round(hr, 4),
        }
    results["best_headroom"] = round(best_headroom, 4)
    results["stage_b_greenlit"] = bool(best_headroom >= 0.05)
    results["honest_verdict"] = (
        "complete: headroom_present_stage_b_greenlit" if best_headroom >= 0.05
        else "complete: no_recoverable_headroom_on_this_baseline"
    )
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[probe] DONE best_headroom={best_headroom:.4f} greenlit={results['stage_b_greenlit']} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
