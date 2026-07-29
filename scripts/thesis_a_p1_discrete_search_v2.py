"""
Thesis-A P1 v2 — discrete-search adjudication, CORRECTED REGIME.

P1 v1 (`thesis_a_p1_discrete_search.py`) was DEGENERATE: it reused the SMALL
matched_compute harness but scaled the task to 3-digit, so the AR positive control
ALSO collapsed (ar=0.000, near-constant "0944" output). With AR=0 you cannot tell
"energy landscape fundamentally misshaped" from "this tiny model learned nothing" —
a positive-control-failed null test (FALSE_NEGATIVE_RISK per CLAUDE.md). v1 is kept
as an honest record with a corrigendum; this v2 supersedes it.

The VALID adjudication regime is the one where AR succeeds and EBT-argmin fails — i.e.
the SCALED harness (`thesis_a_part_b_scaled.py`: dim=768, 4 layers, 12k steps, learned
decoder), which measured AR=0.84 vs EBT-argmin=0.0 (results/thesis_a_part_b_scaled_seed1.json).
v2 reuses THAT exact pipeline, then adds the Deep-Think-P1 decisive probe: replace the
greedy per-token energy-argmin with a GLOBAL discrete beam search minimising cumulative
per-position EBT energy over valid token embeddings.

  - beam >> argmin and approaches AR=0.84  => greedy decode was the bottleneck => ARTIFACT.
  - beam ~= argmin ~= 0  while AR succeeds  => global discrete search ALSO fails => the
    energy landscape itself is misshaped for algorithmic generation => FUNDAMENTAL (causal-
    inductive-bias; closes "EBM generates reasoning" with a reason).
  - GUARD: if AR < 0.3 the run is degenerate (positive control failed) => INCONCLUSIVE,
    NOT fundamental — never repeat the v1 trap.

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


def _load(name, fname):
    spec = _ilu.spec_from_file_location(name, os.path.join(PROJECT_ROOT, "scripts", fname))
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pb = _load("pb", "thesis_a_part_b_matched_compute.py")
sc = _load("sc", "thesis_a_part_b_scaled.py")

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
    cand_emb = emb[cand_ids]
    beams = [(list(pid), 0.0)]
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
            e = ebt(orig, pred)[:, -1, 0]
            nf += VOCAB
            low_e, idx = torch.topk(e, topk, largest=False)
            for j in range(topk):
                expanded.append((ids + [int(cand_ids[int(idx[j])])], cum + float(low_e[j])))
        expanded.sort(key=lambda x: x[1])
        beams = expanded[:beam]
    return beams[0][0][len(pid) :], nf


def main():
    import random

    seed = 30603
    digits, steps, dim, layers, heads = 3, 12000, 768, 4, 12
    K, decoder_steps, n_eval, beam = 30, 1500, 100, 8
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
    print(
        f"[setup] dev={dev} digits={digits} steps={steps} dim={dim} layers={layers} beam={beam}",
        flush=True,
    )

    mu = (10**digits) ** 2
    n_train = min(40000, int(mu * 0.7))  # MUST match the scaled harness (40k) — 20k starves AR
    train_items = pb.build_corpus(digits, n_train, seed)
    tp = {t[0] for t in train_items}
    eval_items = pb.build_corpus(digits, 4000, seed + 777, exclude=tp)
    blocks = pb.corpus_to_blocks(train_items, blk)
    print(
        f"[data] train={len(train_items)} eval={len(eval_items)} blocks={tuple(blocks.shape)}",
        flush=True,
    )

    ebt, ar = sc.build_models(dim, layers, heads, blk, dev)
    import torch.nn as nn

    ckpt = os.path.join(PROJECT_ROOT, "results", "thesis_a_p1_v2_trained.pt")
    dec = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, VOCAB)).to(dev)
    if os.path.exists(ckpt):
        # Resume: training + decoder-fit (the expensive ~100min) already done; reload and
        # skip straight to eval. Lets a re-run after an eval-stage crash be cheap.
        st = torch.load(ckpt, map_location=dev)
        ebt.load_state_dict(st["ebt"])
        ar.load_state_dict(st["ar"])
        dec.load_state_dict(st["dec"])
        nan = st.get("nan", False)
        print(
            f"[resume] loaded trained models from {ckpt} (nan={nan}) — skipping train+decoder",
            flush=True,
        )
    else:
        nan = pb.train_models(
            ebt, ar, blocks, dev, steps, bs=16, langevin=(5, 15), log=lambda m: print(m, flush=True)
        )
        print(f"[train] done nan={nan}", flush=True)
        dec2 = sc.fit_decoder(
            ebt, blocks, dev, K, decoder_steps, log=lambda m: print(m, flush=True)
        )
        dec.load_state_dict(dec2.state_dict())
        torch.save(
            {"ebt": ebt.state_dict(), "ar": ar.state_dict(), "dec": dec.state_dict(), "nan": nan},
            ckpt,
        )
        print(f"[ckpt] saved trained models to {ckpt}", flush=True)

    ans_len = digits + 1
    ebt.eval()
    ar.eval()
    ar1 = arV = argmin = descent = beamc = 0
    arV_nf = 0
    samples = []
    items = eval_items[:n_eval]
    for j, (p, true_ans) in enumerate(items):
        pid = enc(p)
        true = enc(true_ans)
        g1, _ = pb.ar_greedy(ar, pid, ans_len, dev)
        ar1 += g1 == true
        gV, nf = pb.ar_selfconsistency(ar, pid, ans_len, dev, VOCAB)
        arV_nf += nf
        arV += gV == true
        ga, _ = pb.ebt_generate(ebt, pid, ans_len, dev)
        argmin += ga == true
        gd, _ = sc.ebt_descent_generate(ebt, dec, pid, ans_len, dev, K)
        descent += gd == true
        gb, _ = ebt_beam_generate(ebt, pid, ans_len, dev, beam=beam)
        beamc += gb == true
        if j < 8:
            samples.append(
                {
                    "prompt": p,
                    "true": true_ans,
                    "ar1": dec_ids(g1),
                    "ebt_argmin": dec_ids(ga),
                    "ebt_descent": dec_ids(gd),
                    "ebt_beam": dec_ids(gb),
                }
            )
        if (j + 1) % 20 == 0:
            print(
                f"[eval] {j + 1}/{len(items)} ar1={ar1 / (j + 1):.3f} arV={arV / (j + 1):.3f} "
                f"argmin={argmin / (j + 1):.3f} descent={descent / (j + 1):.3f} beam={beamc / (j + 1):.3f}",
                flush=True,
            )
    n = len(items)
    ar1_acc, arV_acc = ar1 / n, arV / n
    am_acc, de_acc, bm_acc = argmin / n, descent / n, beamc / n
    ar_best = max(ar1_acc, arV_acc)
    ebt_best = max(am_acc, de_acc, bm_acc)

    # GUARD: positive control must have succeeded, else INCONCLUSIVE (the v1 trap).
    if ar_best < 0.3:
        verdict = (
            f"complete: thesis_a_p1_v2_INCONCLUSIVE_positive_control_failed_ar_best_{ar_best:.3f}"
            f"_below_0.3_cannot_adjudicate_artifact_vs_fundamental_rerun_with_stronger_ar"
        )
    elif bm_acc > max(0.05, am_acc + 0.05) and bm_acc >= 0.5 * ar_best:
        verdict = (
            f"complete: thesis_a_p1_v2_ARTIFACT_global_beam_rescues_ebt_beam_{bm_acc:.3f}_vs_argmin_{am_acc:.3f}"
            f"_descent_{de_acc:.3f}_ar_{ar_best:.3f}_greedy_decode_was_the_bottleneck_not_the_landscape"
        )
    elif bm_acc <= max(0.05, am_acc + 0.02):
        verdict = (
            f"complete: thesis_a_p1_v2_FUNDAMENTAL_global_beam_ALSO_fails_beam_{bm_acc:.3f}_argmin_{am_acc:.3f}"
            f"_descent_{de_acc:.3f}_while_ar_succeeds_{ar_best:.3f}_energy_landscape_misshaped_for_causal_generation"
            f"_closes_ebm_generates_reasoning"
        )
    else:
        verdict = f"complete: thesis_a_p1_v2_PARTIAL_beam_{bm_acc:.3f}_argmin_{am_acc:.3f}_descent_{de_acc:.3f}_ar_{ar_best:.3f}"

    art = {
        "experiment": "thesis_a_p1_discrete_search_v2",
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "operator_authorized": "2026-06-03 P1 discrete-search adjudication v2 (corrected regime: scaled harness where AR succeeds)",
        "supersedes": "thesis_a_p1_discrete_search (v1 degenerate: positive control AR=0 at 3-digit on the small model)",
        "task": f"{digits}-digit MSD-first addition, scaled harness (dim={dim}/{layers}L, {steps} steps + {decoder_steps}-step learned decoder)",
        "ar_greedy_acc": ar1_acc,
        "ar_selfconsistency_acc": arV_acc,
        "ebt_energy_argmin_acc": am_acc,
        "ebt_learned_decoder_acc": de_acc,
        "ebt_beam_search_acc": bm_acc,
        "ar_best": ar_best,
        "ebt_best": ebt_best,
        "beam_width": beam,
        "training_diverged": nan,
        "n_eval": n,
        "ebt_sample_outputs": samples,
        "model_specs": {
            "dim": dim,
            "n_layers": layers,
            "ebt": "scaled_ebt_from_scratch",
            "ar": "matched",
            "from_scratch": True,
        },
        "device": str(dev),
        "random_seed": seed,
        "reproducibility_checksum": hashlib.sha256(
            json.dumps(
                {
                    "seed": seed,
                    "digits": digits,
                    "steps": steps,
                    "dim": dim,
                    "layers": layers,
                    "beam": beam,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "duration_s": round(time.time() - t0, 2),
    }
    with open(
        os.path.join(PROJECT_ROOT, "results", "thesis_a_p1_discrete_search_v2.json"), "w"
    ) as f:
        json.dump(art, f, indent=2)
    print("\n" + verdict, flush=True)
    print(
        f"[done] ar1={ar1_acc:.3f} arV={arV_acc:.3f} argmin={am_acc:.3f} descent={de_acc:.3f} "
        f"beam={bm_acc:.3f} dur={art['duration_s']}s",
        flush=True,
    )
    print("[samples] " + json.dumps(samples[:5]), flush=True)


if __name__ == "__main__":
    main()
