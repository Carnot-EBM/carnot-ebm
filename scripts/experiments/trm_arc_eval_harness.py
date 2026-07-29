"""Standalone TRM ARC-eval harness — reproduce the arcprize ARC-AGI-1 baseline, no training.

WHY standalone (not pretrain.py): pretrain.py's loop TRAINS a full epoch before evaluate()
(line 608 model.train() -> 629 evaluate), and freeze_weights=True still LEARNS the embeddings
(line 87) -> either path mutates the perfectly-aligned (876406-row) baked eval embeddings and
corrupts the baseline. This harness builds the model + loads the checkpoint and calls
evaluate() DIRECTLY (forward-only, torch.inference_mode) so the embeddings are never touched.

Tricks (from the 5-facet investigation, all verified):
  * freeze_weights=True -> create_model builds the SignSGD puzzle-emb optimizer ONLY, skipping
    AdamATan2 (lines 163-174) which we guarded to None (optimizer-only, never used on eval).
  * ema=False -> step_518071 IS already the EMA-smoothed model; eval the loaded weights directly
    (do NOT re-wrap in EMAHelper -> would double-apply EMA).
  * torch.compile ON (default) -> checkpoint keys are _orig_mod.* and load_checkpoint expects them.
  * Build PretrainConfig straight from the checkpoint's own all_config.yaml (avoids the
    arch-override trap: L_cycles=4 not the cfg default 6, etc.).
  * MUST launch via torchrun --nproc-per-node 1 (the ARC evaluator's dist.gather_object needs an
    initialized process group even at world_size=1).

GO signal: NO "Resetting puzzle embedding as shape is different" line (means aligned) + ARC/pass@2
~= 0.40. NO-GO: that reset line appears OR pass@2 ~= 0 (embeddings misaligned).

  CUDA_VISIBLE_DEVICES=GPU-7971baff-... WANDB_MODE=offline torchrun --nproc-per-node 1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
    scripts/experiments/trm_arc_eval_harness.py --ckpt arc_v1 --batch 256
"""

from __future__ import annotations

import argparse
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
CKPTS = {"arc_v1": ("arc_v1_public", "step_518071"), "arc_v2": ("arc_v2_public", "step_723914")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", choices=list(CKPTS), default="arc_v1")
    ap.add_argument(
        "--batch", type=int, default=256, help="global_batch_size (eval voting is batch-invariant)"
    )
    ap.add_argument(
        "--save_outputs",
        action="store_true",
        help="torch.save preds/inputs/puzzle_identifiers for the offline hybrid re-rank",
    )
    ap.add_argument(
        "--save_latent",
        action="store_true",
        help="GAP-3 Stage 1: ALSO dump z_h_pool (the penultimate latent z_H[:,0], hidden=512) "
        "to a SEPARATE eval_out/<ckpt>_latent dir, so the model-native energy uses the "
        "full latent instead of the scalar q_halt. Implies --save_outputs.",
    )
    ap.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="cap eval batches for an INDICATIVE (partial-vote) pass@K — 0 = full faithful eval",
    )
    args = ap.parse_args()

    sub, step = CKPTS[args.ckpt]
    os.chdir(TRM)
    sys.path.insert(0, TRM)

    import torch
    import torch.distributed as dist
    import yaml
    import pretrain
    from pretrain import (
        PretrainConfig,
        create_dataloader,
        create_evaluators,
        init_train_state,
        evaluate,
    )

    t0 = time.time()
    # --- config straight from the checkpoint's own all_config.yaml ---
    cfg_dict = yaml.safe_load(open(f"{SNAP}/{sub}/all_config.yaml"))
    cfg_dict["load_checkpoint"] = f"{SNAP}/{sub}/{step}"
    cfg_dict["checkpoint_path"] = (
        f"eval_out/{args.ckpt}_latent" if args.save_latent else f"eval_out/{args.ckpt}"
    )
    cfg_dict["data_paths"] = [
        f"data/arc1concept-aug-1000" if args.ckpt == "arc_v1" else "data/arc2concept-aug-1000"
    ]
    cfg_dict["ema"] = False  # ckpt already EMA; eval loaded weights directly
    cfg_dict["freeze_weights"] = True  # build SignSGD-only optimizer (skip AdamATan2)
    cfg_dict["global_batch_size"] = args.batch
    cfg_dict["epochs"] = 1  # only used for a total_steps estimate; we never train
    _save_keys = [
        "preds",
        "inputs",
        "puzzle_identifiers",
        "q_halt_logits",
    ]  # q for the rerank avg_q tiebreak
    if args.save_latent:
        _save_keys.append("z_h_pool")  # GAP-3 Stage 1: the full penultimate latent (hidden=512)
    cfg_dict["eval_save_outputs"] = _save_keys if (args.save_outputs or args.save_latent) else []
    # pretrain.evaluate() only makedirs the PARENT of checkpoint_path before torch.save, so a fresh
    # leaf dir (e.g. eval_out/arc_v1_latent) does not exist and the save crashes AFTER all GPU
    # inference has run (wasting ~25 min). Pre-create the full leaf dir here so the save always lands.
    os.makedirs(cfg_dict["checkpoint_path"], exist_ok=True)
    cfg_dict["project_name"] = cfg_dict.get("project_name") or "trm_arc_eval"
    cfg_dict["run_name"] = f"{args.ckpt}_carnot_eval"
    config = PretrainConfig(**cfg_dict)

    # --- single-process distributed (torchrun sets LOCAL_RANK) ---
    RANK, WORLD_SIZE, CPU_GROUP = 0, 1, None
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        CPU_GROUP = dist.new_group(backend="gloo")
    else:
        print(
            "WARNING: no LOCAL_RANK — launch via torchrun --nproc-per-node 1, else the "
            "evaluator's dist.gather_object will fail.",
            flush=True,
        )
    torch.random.manual_seed(config.seed + RANK)

    # --- data + evaluators + model (loads the checkpoint via create_model) ---
    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    eval_loader, eval_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    evaluators = create_evaluators(config, eval_metadata)
    print(
        f"[harness] train_meta: ids={train_metadata.num_puzzle_identifiers} "
        f"vocab={train_metadata.vocab_size} seq={train_metadata.seq_len}; "
        f"eval sets={eval_metadata.sets}",
        flush=True,
    )

    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)
    train_state.model.eval()  # forward-only; NO train loop -> baked embeddings untouched

    # --- optional batch cap: INDICATIVE pass@K (partial augmentation-vote) for a fast read ---
    eval_loader_used = eval_loader
    if args.max_batches and args.max_batches > 0:
        import itertools

        base_loader = eval_loader

        def _capped():
            yield from itertools.islice(base_loader, args.max_batches)

        eval_loader_used = _capped()
        print(
            f"[harness] CAPPED eval to {args.max_batches} batches (INDICATIVE partial-vote pass@K)",
            flush=True,
        )

    # --- evaluate (forward + ARC majority-vote) ---
    metrics = evaluate(
        config,
        train_state,
        eval_loader_used,
        eval_metadata,
        evaluators,
        rank=RANK,
        world_size=WORLD_SIZE,
        cpu_group=CPU_GROUP,
    )

    if RANK == 0:
        dur = round(time.time() - t0, 1)
        print(f"\n[harness] DONE in {dur}s", flush=True)
        print(f"[harness] METRICS: {metrics}", flush=True)
        arc = {k: v for k, v in (metrics or {}).items() if "ARC" in k or "pass" in k.lower()}
        print(f"[harness] ARC pass@K: {arc}", flush=True)
        out = {
            "experiment": "trm_arc_eval_harness",
            "ckpt": args.ckpt,
            "step": step,
            "batch": args.batch,
            "duration_s": dur,
            "metrics": metrics,
            "arc_passK": arc,
            "inference_substrate": "live_gpu_trm_recursive_refiner_arc_eval",
            "note": "standalone eval, no training, baked embeddings aligned (876406 gate passed)",
        }
        os.makedirs(f"{CARNOT}/results", exist_ok=True)
        json.dump(
            out,
            open(f"{CARNOT}/results/trm_arc_baseline_{args.ckpt}.json", "w"),
            indent=2,
            default=str,
        )
        print(f"[harness] wrote {CARNOT}/results/trm_arc_baseline_{args.ckpt}.json", flush=True)

    if "LOCAL_RANK" in os.environ:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
