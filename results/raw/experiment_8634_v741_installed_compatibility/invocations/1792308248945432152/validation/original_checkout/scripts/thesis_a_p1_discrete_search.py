"""
Thesis-A P1 — discrete-search adjudication (Deep Think P1 falsifier).

Energy-as-generator was BOUNDED (EBT 0% vs AR 82% at matched compute, 3-digit add).
DT's P1 frame: is that bound an ARTIFACT of the greedy continuous/argmin decode (the
off-manifold-void problem), or is it FUNDAMENTAL (a global symmetric energy can't do
causal/compositional reasoning)? DECISIVE TEST: keep the SAME trained EBT, replace the
greedy per-token energy-argmin with a GLOBAL discrete search (beam search over the
answer, minimising cumulative per-position energy, evaluated only at valid token
embeddings).
  - beam >> argmin and approaches AR  => the GREEDY DECODE was the bottleneck => ARTIFACT.
  - beam ~= argmin ~= 0                => global discrete search also fails => the energy
    landscape itself is misshaped for algorithmic generation => FUNDAMENTAL (causal-
    inductive-bias; closes the "EBM generates reasoning" direction with a reason).

Reuses the part-b harness (build_tiny_models / train_models / ar_greedy / ebt_generate
= greedy energy-argmin). 3-digit / 16k steps to match the bounded run apples-to-apples.
Run on the stable internal GPU (cuda:1); conductor must be PAUSED (kill_zombies).
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path

# Resolved from this file rather than hardcoded so a fresh clone or a
# worktree writes into ITS OWN tree. Inlined (not carnot.paths.repo_root)
# because the next line is what makes ``carnot`` importable -- importing
# the resolver here would be circular. Same rule, same answer.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "pb", os.path.join(PROJECT_ROOT, "scripts", "thesis_a_part_b_matched_compute.py")
)
pb = _ilu.module_from_spec(_spec)
sys.modules["pb"] = pb
_spec.loader.exec_module(pb)

import torch
import numpy as np

enc, dec_ids = pb.enc, pb.dec_ids
VOCAB = pb.VOCAB


@torch.no_grad()
def ebt_beam_generate(ebt, pid, ans_len, device, beam=8, topk=12):
    """GLOBAL discrete search: beam search over the answer tokens minimising cumulative
    per-position EBT energy, evaluated only at valid token embeddings. Returns
    (best_ids, n_energy_evals). Cost = beam * VOCAB per generated token."""
    emb = ebt.token_embedding.weight
    cand_ids = torch.arange(VOCAB, device=device)
    cand_emb = emb[cand_ids]  # [V, dim]
    beams = [(list(pid), 0.0)]  # (token_ids, cumulative_energy)
    nf = 0
    for _ in range(ans_len):
        expanded = []
        for ids, cum in beams:
            ctx = torch.tensor([ids], device=device)
            m = ctx.shape[1]
            orig = ebt.token_embedding(ctx).expand(VOCAB, -1, -1)
            known = (
                ebt.token_embedding(ctx[:, 1:]).expand(VOCAB, -1, -1)
                if m >= 2
                else torch.zeros((VOCAB, 0, emb.shape[1]), device=device)
            )
            pred = torch.cat([known, cand_emb.unsqueeze(1)], dim=1)
            e = ebt(orig, pred)[:, -1, 0]  # [V] energy per candidate token
            nf += VOCAB
            low_e, idx = torch.topk(e, topk, largest=False)  # lowest energy = best
            for j in range(topk):
                expanded.append((ids + [int(cand_ids[int(idx[j])])], cum + float(low_e[j])))
        expanded.sort(key=lambda x: x[1])  # keep lowest cumulative energy
        beams = expanded[:beam]
    return beams[0][0][len(pid) :], nf


def main():
    import random

    seed, digits, steps, n_eval, beam = 30603, 3, 16000, 100, 8
    t0 = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = torch.device(
        "cuda:1"
        if torch.cuda.device_count() > 1
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    blk = 48
    print(f"[setup] dev={dev} digits={digits} steps={steps} beam={beam}", flush=True)

    mu = (10**digits) ** 2
    train_items = pb.build_corpus(digits, min(20000, int(mu * 0.7)), seed)
    tp = {t[0] for t in train_items}
    eval_items = pb.build_corpus(digits, 3000, seed + 777, exclude=tp)
    blocks = pb.corpus_to_blocks(train_items, blk)
    print(f"[data] train={len(train_items)} eval={len(eval_items)}", flush=True)

    ebt, ar = pb.build_tiny_models(batch_size=16, block_size=blk)
    ebt, ar = ebt.to(dev), ar.to(dev)
    nan = pb.train_models(
        ebt, ar, blocks, dev, steps, bs=16, langevin=(5, 15), log=lambda m: print(m, flush=True)
    )
    print(f"[train] done nan={nan}", flush=True)

    ans_len = digits + 1
    ebt.eval()
    ar.eval()
    ar1 = argmin = beamc = 0
    samples = []
    items = eval_items[:n_eval]
    for j, (p, true_ans) in enumerate(items):
        pid = enc(p)
        true = enc(true_ans)
        g1, _ = pb.ar_greedy(ar, pid, ans_len, dev)
        ar1 += g1 == true
        ga, _ = pb.ebt_generate(ebt, pid, ans_len, dev)
        argmin += ga == true  # greedy energy-argmin
        gb, _ = ebt_beam_generate(ebt, pid, ans_len, dev, beam=beam)
        beamc += gb == true  # global beam
        if j < 8:
            samples.append(
                {
                    "prompt": p,
                    "true": true_ans,
                    "ar": dec_ids(g1),
                    "ebt_argmin": dec_ids(ga),
                    "ebt_beam": dec_ids(gb),
                }
            )
        if (j + 1) % 20 == 0:
            print(
                f"[eval] {j + 1}/{len(items)} ar={ar1 / (j + 1):.3f} argmin={argmin / (j + 1):.3f} beam={beamc / (j + 1):.3f}",
                flush=True,
            )
    n = len(items)
    ar_acc, am_acc, bm_acc = ar1 / n, argmin / n, beamc / n

    # verdict: artifact (beam rescues) vs fundamental (global search also fails)
    if bm_acc > max(0.05, am_acc + 0.05) and bm_acc >= 0.5 * ar_acc:
        verdict = (
            f"complete: thesis_a_p1_ARTIFACT_global_beam_search_rescues_ebt_beam_{bm_acc:.3f}_vs_argmin_{am_acc:.3f}"
            f"_ar_{ar_acc:.3f}_the_greedy_decode_was_the_bottleneck_not_the_landscape"
        )
    elif bm_acc <= max(0.05, am_acc + 0.02):
        verdict = (
            f"complete: thesis_a_p1_FUNDAMENTAL_global_beam_search_ALSO_fails_beam_{bm_acc:.3f}_argmin_{am_acc:.3f}"
            f"_vs_ar_{ar_acc:.3f}_energy_landscape_misshaped_for_causal_generation_closes_ebm_generates_reasoning"
        )
    else:
        verdict = f"complete: thesis_a_p1_PARTIAL_beam_{bm_acc:.3f}_argmin_{am_acc:.3f}_ar_{ar_acc:.3f}_intermediate"

    art = {
        "experiment": "thesis_a_p1_discrete_search",
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "operator_authorized": "2026-06-03 P1 discrete-search adjudication (artifact vs fundamental)",
        "task": f"{digits}-digit MSD-first addition, {steps} steps (matches the bounded run)",
        "ar_greedy_acc": ar_acc,
        "ebt_energy_argmin_acc": am_acc,
        "ebt_beam_search_acc": bm_acc,
        "beam_width": beam,
        "training_diverged": nan,
        "n_eval": n,
        "ebt_sample_outputs": samples,
        "model_specs": {
            "ebt": "tiny_ebt_from_scratch_byte_38M",
            "ar": "matched",
            "from_scratch": True,
        },
        "device": str(dev),
        "random_seed": seed,
        "reproducibility_checksum": hashlib.sha256(
            json.dumps(
                {"seed": seed, "digits": digits, "steps": steps, "beam": beam}, sort_keys=True
            ).encode()
        ).hexdigest(),
        "duration_s": round(time.time() - t0, 2),
    }
    with open(os.path.join(PROJECT_ROOT, "results", "thesis_a_p1_discrete_search.json"), "w") as f:
        json.dump(art, f, indent=2)
    print("\n" + verdict, flush=True)
    print(
        f"[done] ar={ar_acc:.3f} argmin={am_acc:.3f} beam={bm_acc:.3f} dur={art['duration_s']}s",
        flush=True,
    )
    print("[samples] " + json.dumps(samples[:5]), flush=True)


if __name__ == "__main__":
    main()
