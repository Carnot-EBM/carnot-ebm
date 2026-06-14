#!/usr/bin/env python3
"""Verifier-as-DETECTOR AUROC probe (operator-chosen direction, 2026-06-14).

Isolates the verifier's DETECTION value (can it flag/reject wrong outputs?) from its
SELECTION value (oracle@K-vote, headroom-conditional). On Sudoku the executable verifier
(constraint satisfaction) should separate correct from incorrect outputs with high AUROC
EVEN at the converged checkpoint where selection headroom was ~0 (exp v3 headroom curve) --
the headline contrast: detection works where selection can't.

HONEST framing (Circularity Discipline): on Sudoku the verifier IS the executable oracle
(constraint check ~ correctness for a unique-solution puzzle), so this is an
EXECUTION_GROUNDED / verifier_is_oracle=TRUE detection result -- valid, but not a
headline moat. The point is the detection-vs-selection DIVERGENCE, plus the abstention
curve (accuracy vs coverage), which is the actually-useful capability (precision / "I don't
know"). The non-circular detector (diffusion-surprisal / a learned detector) is future work.

Output: results/verifier_detector_auroc.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_orig_load = torch.load
torch.load = lambda *a, **k: (k.update(weights_only=False) or _orig_load(*a, **k))  # trusted ckpt

NANO_TRM = "/home/ianblenke/github.com/ianblenke/carnot/nano-trm"
sys.path.insert(0, NANO_TRM)
from src.nn.sudoku_evaluator import SudokuEvaluator  # noqa: E402
from src.nn.utils.constants import IGNORE_LABEL_ID  # noqa: E402

STABLE = "/home/ianblenke/github.com/ianblenke/carnot/results/trm_runs/sudoku_extreme_baseline/last.ckpt"
DATA_DIR = f"{NANO_TRM}/data/sudoku_extreme_1k_aug_1k"
OUT = Path("/home/ianblenke/github.com/ianblenke/carnot/results/verifier_detector_auroc.json")


def constraint_sat_fraction(grid: torch.Tensor, n: int = 9) -> float:
    """Fraction of row/col/box all-distinct-1..n constraints satisfied. grid: [n,n] ints."""
    box = 3
    total = 3 * n  # n rows + n cols + n boxes
    ok = 0
    for r in range(n):
        row = grid[r]
        if torch.all((row >= 1) & (row <= n)) and len(torch.unique(row)) == n:
            ok += 1
    for c in range(n):
        col = grid[:, c]
        if torch.all((col >= 1) & (col <= n)) and len(torch.unique(col)) == n:
            ok += 1
    for br in range(0, n, box):
        for bc in range(0, n, box):
            b = grid[br:br + box, bc:bc + box].reshape(-1)
            if torch.all((b >= 1) & (b <= n)) and len(torch.unique(b)) == n:
                ok += 1
    return ok / total


def auroc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUROC (Mann-Whitney). labels: 1=positive(correct)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    return (sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


@torch.no_grad()
def main() -> None:
    t0 = time.time()
    ev = SudokuEvaluator(checkpoint_path=STABLE, data_dir=DATA_DIR, batch_size=128,
                         device="auto", eval_split="test")
    ev.datamodule.setup("test")
    loader = ev.datamodule.test_dataloader()
    m = ev.model
    m.eval()
    n = 9

    # --- decoding sanity check: gold labels must be valid Sudoku (confirms token==digit) ---
    sanity = {"gold_valid_frac": None}
    first = next(iter(loader))
    lab0 = first["output"][:16]
    gv = 0
    for i in range(lab0.shape[0]):
        g = lab0[i].reshape(n, n)
        if constraint_sat_fraction(g, n) == 1.0:
            gv += 1
    sanity["gold_valid_frac"] = gv / lab0.shape[0]

    scores, labels, valid_flags = [], [], []
    greedy_correct = 0
    total = 0
    for batch in loader:
        b = {k: v.to(ev.device) for k, v in batch.items()}
        carry = m.initial_carry(b)
        steps = 0
        logits = None
        while True:
            carry, out = m.forward(carry, b)
            logits = out["logits"]
            steps += 1
            if carry.halted.all() or steps > 64:
                break
        pred = logits.argmax(-1)            # [B, 81]
        label = carry.current_data["output"]
        mask = label != IGNORE_LABEL_ID
        B = label.shape[0]
        for i in range(B):
            p = pred[i]
            lab = label[i]
            mk = mask[i]
            is_corr = bool(((p == lab) | (~mk)).all().item())
            grid = p.reshape(n, n).clamp(0, n)  # decode tokens->digits (identity per sanity)
            vscore = constraint_sat_fraction(grid, n)
            scores.append(vscore)
            labels.append(1 if is_corr else 0)
            valid_flags.append(vscore == 1.0)
            greedy_correct += int(is_corr)
            total += 1

    det_auroc = auroc(scores, labels)
    base_rate = sum(labels) / max(1, len(labels))
    # valid-but-wrong hard split: among outputs the model thinks are fully valid, can the
    # (continuous) score still separate? (here exact_valid is near-binary, so report the
    # subset where it's a HARD call: outputs that are NOT exact-valid -> does score rank them?)
    hard_idx = [i for i in range(len(scores)) if not valid_flags[i]]
    hard_auroc = auroc([scores[i] for i in hard_idx], [labels[i] for i in hard_idx]) if hard_idx else float("nan")
    # abstention curve: accuracy if we reject the lowest-score k% (coverage)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    abst = []
    for cov in (1.0, 0.9, 0.75, 0.5, 0.25):
        keep = order[: int(len(order) * cov)] or order[:1]
        acc = sum(labels[i] for i in keep) / len(keep)
        abst.append({"coverage": cov, "accuracy": round(acc, 4)})

    rep = {
        "experiment": "verifier_detector_auroc",
        "inference_substrate": "verifier_ensemble_against_live_trm_outputs",
        "verifier_is_oracle": True,
        "verifier_is_oracle_note": "Sudoku constraint-check ~ correctness (unique solution) -> EXECUTION_GROUNDED/circular detection; valid result, NOT a headline moat. The non-circular detector (diffusion-surprisal/learned) is future work.",
        "checkpoint": STABLE,
        "n": len(labels),
        "greedy_exact_accuracy": round(greedy_correct / max(1, total), 4),
        "gold_decoding_sanity": sanity,
        "detection_auroc": round(det_auroc, 4),
        "base_rate_correct": round(base_rate, 4),
        "hard_split_auroc_nonvalid_outputs": round(hard_auroc, 4) if hard_auroc == hard_auroc else None,
        "abstention_curve": abst,
        "duration_s": round(time.time() - t0, 1),
        "honest_verdict": "complete: detector_auroc_measured_execution_grounded_circular",
    }
    OUT.write_text(json.dumps(rep, indent=2))
    print(f"[detector] DONE auroc={det_auroc:.4f} greedy={rep['greedy_exact_accuracy']} "
          f"gold_sanity={sanity['gold_valid_frac']} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
