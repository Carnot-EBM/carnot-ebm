"""GAP-3 Stage 2 v2 — mine REAL TRM wrong candidates from the TRAINING split as hard negatives.

WHY (the Stage-2 v1 adversarial round, results/arc3_gap3_stage2_adversarial_verify.json): v1's synthetic
corruption negatives cover only 3.8% of TRM's real wrong-candidate distribution; 91.5% of real errors
are same-shape near-misses / plausible-but-wrong rule applications the synthetic families never produce.
The panel's recommended fix #1: train against REAL generator errors mined from TRAINING-split tasks only.

WHAT THIS DOES: runs the TRM arc_v1 checkpoint forward (inference only, frozen weights, EMA-baked) over
a CAPPED slice of its own baked TRAINING split (data/arc1concept-aug-1000/train — ARC-1 training +
ConceptARC tasks ONLY; the eval tasks live in the /test split), saves (inputs, labels, preds,
puzzle_identifiers), then de-augments every row and keeps the predictions whose grid differs from the
row's own gold label. Those wrong predictions ARE TRM's real error distribution, harvested with ZERO
eval-task leak (hard-asserted against the eval pool).

HONESTY NOTE (disclosed in the v2 artifact): mining from the SAME checkpoint that generates the eval
candidates forfeits STRICT generator-independence — the v2 energy is tuned to TRM's error distribution.
Task hygiene is fully preserved (training-split tasks only; the de-aug join uses each row's own label,
never an eval solution). The panel pre-approved exactly this trade (reviewer 3: "mine TRM wrong
candidates from TRAINING-split tasks — keeps eval-task hygiene, drops strict generator-independence").

  # conductor must be paused (its preflight reaps non-conductor GPU procs); ~35 min for 120 batches:
  CUDA_VISIBLE_DEVICES=GPU-7971baff-... WANDB_MODE=offline ~/trm_venv/bin/torchrun \
    --nproc-per-node 1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
    scripts/experiments/arc3_gap3_stage2v2_mine_real_negs.py --batch 256 --max_batches 120
  # then the CPU join step:
  ~/trm_venv/bin/python scripts/experiments/arc3_gap3_stage2v2_mine_real_negs.py --join_only

CORRIGENDUM (2026-06-09, post-adversarial-round — see corrigendum_2026_06_09_stage2v2 in the v2
artifact): this miner has TWO known bugs that made the mined axis a silent no-op, documented here
because the LINEAGE IS RETIRED and the script is preserved as evidence, not fixed for reuse:
  1. inv_fn(_crop(inputs[i])) yields EMPTY grids on translation-augmented rows (the evaluator's _crop
     was built for untranslated test rows) -> 240/250 mined negatives were keyed to the empty-grid
     hash and could never join a training example.
  2. The wrong-row rate (59.9%) was a padding artifact off ~500x: 18,380/18,390 'wrong rows' are
     padding rows. TRM's GENUINE error rate on real rows of its own training split is 10/8,259 = 0.12%
     (it memorizes the split) — so even a fixed miner cannot supply a real-error-dominant curriculum
     from this checkpoint's training split.
If mining is ever revived for a DIFFERENT lineage: keep a row only if its de-augmented label
content-matches the kaggle gold, assert the recovered input is non-empty, and prefer an early or
independent checkpoint whose train-split errors are not memorization-suppressed.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time
from carnot.paths import repo_root

TRM = "/home/ianblenke/trm_src"
SNAP = (
    "/home/ianblenke/.cache/huggingface/hub/models--arcprize--trm_arc_prize_verification/"
    "snapshots/55ced5dd59de74c52f53d47aa2898232b5a15b7a"
)
# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
CARNOT = str(repo_root())
SUB, STEP = "arc_v1_public", "step_518071"
DUMP_DIR = f"{TRM}/eval_out/arc_v1_trainmine"
OUT = f"{CARNOT}/results/arc3_gap3_stage2v2_mined_negs.json.gz"


def run_dump(args):
    os.chdir(TRM)
    sys.path.insert(0, TRM)
    import itertools

    import torch
    import torch.distributed as dist
    import yaml
    from pretrain import PretrainConfig, create_dataloader, evaluate, init_train_state

    t0 = time.time()
    cfg_dict = yaml.safe_load(open(f"{SNAP}/{SUB}/all_config.yaml"))
    cfg_dict["load_checkpoint"] = f"{SNAP}/{SUB}/{STEP}"
    cfg_dict["checkpoint_path"] = "eval_out/arc_v1_trainmine"
    cfg_dict["data_paths"] = ["data/arc1concept-aug-1000"]
    cfg_dict["ema"] = False
    cfg_dict["freeze_weights"] = True
    cfg_dict["global_batch_size"] = args.batch
    cfg_dict["epochs"] = 1
    # labels is the per-row gold — saving it makes the wrong-pred join self-contained (no kaggle join)
    cfg_dict["eval_save_outputs"] = ["inputs", "labels", "preds", "puzzle_identifiers"]
    os.makedirs(cfg_dict["checkpoint_path"], exist_ok=True)
    cfg_dict["project_name"] = cfg_dict.get("project_name") or "trm_arc_eval"
    cfg_dict["run_name"] = "arc_v1_trainmine"
    config = PretrainConfig(**cfg_dict)

    RANK, WORLD_SIZE, CPU_GROUP = 0, 1, None
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        CPU_GROUP = dist.new_group(backend="gloo")
    torch.random.manual_seed(config.seed + RANK)

    # train_metadata for model init (embedding-table alignment), TRAIN split in TEST-set mode for the
    # sequential capped inference pass — test_set_mode switches the iterator, not the data.
    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    mine_loader, mine_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)
    train_state.model.eval()

    capped = mine_loader
    if args.max_batches:
        base = mine_loader

        def _capped():
            yield from itertools.islice(base, args.max_batches)

        capped = _capped()
        print(f"[miner] CAPPED to {args.max_batches} batches of the TRAIN split", flush=True)

    # evaluators=[] -> pure forward + save; no ARC evaluator (it joins against EVAL test_puzzles,
    # meaningless for train-split rows)
    evaluate(
        config,
        train_state,
        capped,
        mine_metadata,
        [],
        rank=RANK,
        world_size=WORLD_SIZE,
        cpu_group=CPU_GROUP,
    )
    if RANK == 0:
        print(f"[miner] dump DONE in {time.time() - t0:.0f}s -> {DUMP_DIR}", flush=True)
    if "LOCAL_RANK" in os.environ:
        dist.destroy_process_group()


def run_join():
    """CPU: de-augment every dumped row; keep predictions that differ from the row's own gold label.
    Output: {task_name: {input_hash: {input: grid, negs: [grids]}}} + stats."""
    sys.path.insert(0, TRM)
    sys.path.insert(0, f"{CARNOT}/scripts/experiments")
    import numpy as np
    import torch
    from dataset.build_arc_dataset import grid_hash, inverse_aug
    from evaluators.arc import ARC, _crop
    from types import SimpleNamespace

    from arc3_gap3_stage2_transition_ebm import ghash

    t0 = time.time()
    ev = ARC(
        data_path=f"{TRM}/data/arc1concept-aug-1000",
        eval_metadata=SimpleNamespace(blank_identifier_id=0),
    )
    shards = sorted(glob.glob(f"{DUMP_DIR}/step_0_all_preds.*"))
    assert shards, f"no dump at {DUMP_DIR} — run the torchrun dump step first"

    store = {}
    n_rows = n_wrong = 0
    for sh in shards:
        d = torch.load(sh, map_location="cpu")
        pids = d["puzzle_identifiers"].numpy()
        inputs = d["inputs"].numpy()
        labels = d["labels"].numpy()
        preds = d["preds"].numpy()
        keep = pids != ev.blank_identifier_id
        for i in np.nonzero(keep)[0]:
            name = ev.identifier_map[int(pids[i])]
            orig_name, inv_fn = inverse_aug(name)
            n_rows += 1
            gold = inv_fn(_crop(labels[i]))
            pred = inv_fn(_crop(preds[i]))
            if grid_hash(pred) == grid_hash(gold):
                continue
            n_wrong += 1
            tin = inv_fn(_crop(inputs[i]))
            ih = ghash(np.asarray(tin).astype(int))
            slot = store.setdefault(orig_name, {}).setdefault(
                ih, {"input": np.asarray(tin).astype(int).tolist(), "negs": {}}
            )
            ph = ghash(np.asarray(pred).astype(int))
            if ph not in slot["negs"] and len(slot["negs"]) < 24:
                slot["negs"][ph] = np.asarray(pred).astype(int).tolist()

    # hard split-hygiene assertions
    pool_tasks = set()
    with gzip.open(f"{CARNOT}/results/arc3_gap3_stage2_eval_pool.json.gz", "rt") as f:
        for e in json.load(f)["entries"]:
            pool_tasks.add(e["task"])
    leak = pool_tasks & set(store)
    assert not leak, f"EVAL LEAK in mined negatives: {leak}"
    ch = json.load(open("/home/ianblenke/trm_src/kaggle/combined/arc-agi_training_challenges.json"))
    cc = json.load(open("/home/ianblenke/trm_src/kaggle/combined/arc-agi_concept_challenges.json"))
    train_names = set(ch) | set(cc)
    stray = set(store) - train_names
    assert not stray, f"mined tasks outside the training corpus: {stray}"

    out = {
        "experiment": "arc3_gap3_stage2v2_mined_negs",
        "n_rows_scanned": n_rows,
        "n_wrong_rows": n_wrong,
        "n_tasks_with_negs": len(store),
        "n_distinct_negs": sum(len(s["negs"]) for t in store.values() for s in t.values()),
        "generator": f"TRM {SUB}/{STEP} on its own TRAINING split (zero eval-task leak, asserted)",
        "tasks": {
            t: {
                ih: {"input": s["input"], "negs": list(s["negs"].values())}
                for ih, s in slots.items()
            }
            for t, slots in store.items()
        },
    }
    with gzip.open(OUT, "wt") as f:
        json.dump(out, f)
    print(
        f"[join] {n_rows} rows scanned, {n_wrong} wrong ({n_wrong / max(1, n_rows):.1%}), "
        f"{out['n_distinct_negs']} distinct negs across {len(store)} tasks "
        f"in {time.time() - t0:.0f}s -> {OUT}",
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max_batches", type=int, default=120)
    ap.add_argument("--join_only", action="store_true")
    a = ap.parse_args()
    if a.join_only:
        run_join()
    else:
        run_dump(a)
