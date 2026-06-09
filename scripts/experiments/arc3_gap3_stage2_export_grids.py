"""GAP-3 Stage 2: export the eval pool WITH GRIDS — per eval entry the task's demo pairs, the test
input grid, and every TRM candidate grid (de-augmented) with votes/q_mean/correct labels.

WHY: Stage 2 is the trained generator-INDEPENDENT ARC transition-EBM. Its energy is computed from grid
CONTENT — E(candidate | test_input, demo_pairs) — so the eval harness needs the actual grids, which the
Stage-1 scalar/latent table (results/arc3_gap3_stage1_candidate_table.{npz,json}) does not carry. This
exporter replays the exact same de-aug join as Stage 1 (same dump, same bucket structure, byte-identical
votes/q_mean/correct columns) and attaches:
  * demos: the task's train pairs (from test_puzzles.json — these are the DEMO pairs, public context)
  * test_input: the test input grid the candidates answer
  * candidates[].grid: the de-augmented predicted output grid
NO-ORACLE NOTE: the gold test OUTPUT grid is read ONLY to compute the boolean `correct` label (used for
SCORING, never as a model input) and is NOT exported. The energy at inference reads demos + test_input +
candidate grid only (REQ-GAP3-2).

  ~/trm_venv/bin/python scripts/experiments/arc3_gap3_stage2_export_grids.py
"""

from __future__ import annotations

import glob
import gzip
import json
import sys
from types import SimpleNamespace

import numpy as np

CARNOT = "/home/ianblenke/github.com/ianblenke/carnot"
TRM = "/home/ianblenke/trm_src"
sys.path.insert(0, TRM)

DATA = f"{TRM}/data/arc1concept-aug-1000"
LATENT_GLOB = f"{TRM}/eval_out/arc_v1_latent/step_0_all_preds.*"

import torch  # noqa: E402
from evaluators.arc import ARC, _crop  # noqa: E402
from dataset.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np  # noqa: E402


def main():
    ev = ARC(data_path=DATA, eval_metadata=SimpleNamespace(blank_identifier_id=0))
    shards = sorted(glob.glob(LATENT_GLOB))
    assert shards, f"no latent dump at {LATENT_GLOB}"
    # accumulate per (orig_task, input_hash): {pred_hash: {votes, qs[], grid}} — same join as Stage 1,
    # but carrying the de-augmented predicted GRID instead of the latent.
    store = {}
    for sh in shards:
        d = torch.load(sh, map_location="cpu")
        pids = d["puzzle_identifiers"].numpy()
        inputs = d["inputs"].numpy()
        preds = d["preds"].numpy()
        qh = d["q_halt_logits"].to(torch.float64).sigmoid().numpy()
        keep = pids != ev.blank_identifier_id
        for i in np.nonzero(keep)[0]:
            name = ev.identifier_map[int(pids[i])]
            orig_name, inv_fn = inverse_aug(name)
            ih = grid_hash(inv_fn(_crop(inputs[i])))
            pg = inv_fn(_crop(preds[i]))
            ph = grid_hash(pg)
            slot = (
                store.setdefault(orig_name, {})
                .setdefault(ih, {})
                .setdefault(ph, {"votes": 0, "qs": [], "grid": None})
            )
            slot["votes"] += 1
            slot["qs"].append(float(qh[i]))
            if slot["grid"] is None:
                slot["grid"] = np.asarray(pg).astype(int).tolist()

    entries = []
    for name, puzzle in ev.test_puzzles.items():
        if name not in store:
            continue
        for pair in puzzle["test"]:
            ih = grid_hash(arc_grid_to_np(pair["input"]))
            lh = grid_hash(arc_grid_to_np(pair["output"]))  # gold hash — LABEL ONLY, grid not exported
            bucket = store[name].get(ih)
            if not bucket:
                continue
            cands = [
                {
                    "votes": s["votes"],
                    "q_mean": float(np.mean(s["qs"])),
                    "correct": bool(ph == lh),
                    "grid": s["grid"],
                }
                for ph, s in bucket.items()
            ]
            if cands:
                entries.append(
                    {
                        "task": name,
                        "demos": puzzle["train"],  # public demo pairs (context, not oracle)
                        "test_input": pair["input"],
                        "candidates": cands,
                    }
                )

    out = {
        "experiment": "arc3_gap3_stage2_eval_pool",
        "n_entries": len(entries),
        "n_candidates": sum(len(e["candidates"]) for e in entries),
        "note": (
            "Eval pool for the Stage-2 transition-EBM rerank. Same de-aug join as Stage 1 "
            "(byte-identical votes/q_mean/correct); adds demo pairs + test_input + candidate grids. "
            "Gold test outputs are NOT exported (no-oracle: REQ-GAP3-2); `correct` is a scoring label."
        ),
        "entries": entries,
    }
    path = f"{CARNOT}/results/arc3_gap3_stage2_eval_pool.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(out, f)
    # cross-check against the Stage-1 sidecar: per-entry candidate count + vote multiset must match
    s1 = json.load(open(f"{CARNOT}/results/arc3_gap3_stage1_candidate_table.json"))
    assert len(entries) == s1["n_tasks"], (len(entries), s1["n_tasks"])
    s1_votes = {}
    for r in s1["candidates"]:
        s1_votes.setdefault(r["task_idx"], []).append(r["votes"])
    for ti, e in enumerate(entries):
        a = sorted(c["votes"] for c in e["candidates"])
        b = sorted(s1_votes[ti])
        assert a == b, f"entry {ti} vote multiset mismatch"
    print(
        f"exported {len(entries)} entries / {out['n_candidates']} candidates -> {path} "
        f"(cross-checked vs Stage-1 sidecar: OK)"
    )


if __name__ == "__main__":
    main()
