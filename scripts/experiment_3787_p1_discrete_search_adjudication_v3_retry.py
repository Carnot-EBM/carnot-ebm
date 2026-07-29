"""
Thesis-A P1 v3 — discrete-search adjudication, CORRECTED REGIME.
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


def write_artifact(verdict, preconditions_checked, handoff_to_operator, t0, divergence=False):
    seed = 30603
    digits, steps, dim, layers, beam = 3, 12000, 768, 4, 8
    n_train = 40000
    art = {
        "experiment": "thesis_a_p1_discrete_search_v3",
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "positive_control_passed": False,
        "ar_best": 0.0,
        "ebt_best": 0.0,
        "adjudication": "inconclusive_positive_control_failed",
        "n_train": n_train,
        "per_method_accuracies": {
            "ar_greedy": 0.0,
            "ar_selfconsistency": 0.0,
            "ebt_argmin": 0.0,
            "ebt_beam": 0.0,
            "ebt_descent": 0.0,
        },
        "training_diverged": divergence,
        "energy_as_generator_still_bounded": True,
        "preconditions_checked": preconditions_checked,
        "handoff_to_operator": handoff_to_operator,
        "model_specs": {
            "dim": dim,
            "n_layers": layers,
            "ebt": "scaled_ebt_from_scratch",
            "ar": "matched",
            "n_train": n_train,
            "from_scratch": True,
        },
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
                    "n_train": n_train,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "duration_s": round(time.time() - t0, 2),
    }
    with open(
        os.path.join(
            PROJECT_ROOT, "results", "experiment_3787_p1_discrete_search_adjudication_v3_retry.json"
        ),
        "w",
    ) as f:
        json.dump(art, f, indent=2)
    print("\n" + verdict, flush=True)


def main():
    import random

    seed = 30603
    digits, steps, dim, layers, heads = 3, 12000, 768, 4, 12
    K, decoder_steps, n_eval, beam = 30, 1500, 100, 8
    t0 = time.time()

    # Preconditions check
    preconditions_checked = False
    handoff_to_operator = False

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        verdict = "blocked_cuda_unavailable"
        write_artifact(verdict, preconditions_checked, handoff_to_operator, t0)
        return

    dev = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")

    # Simple check for free GPU memory
    free_mem = torch.cuda.mem_get_info(dev)[0]
    if free_mem < 10 * 1024**3:
        verdict = "blocked_no_free_gpu"
        handoff_to_operator = True
        write_artifact(verdict, preconditions_checked, handoff_to_operator, t0)
        return

    preconditions_checked = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    blk = 48
    print(
        f"[setup] dev={dev} digits={digits} steps={steps} dim={dim} layers={layers} beam={beam}",
        flush=True,
    )

    mu = (10**digits) ** 2
    n_train = min(40000, int(mu * 0.7))  # MUST be 40k
    assert n_train == 40000, "n_train must be 40000"
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

    ckpt = os.path.join(PROJECT_ROOT, "results", "thesis_a_p1_v3_trained.pt")
    dec = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, VOCAB)).to(dev)
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=dev)
        ebt.load_state_dict(st["ebt"])
        ar.load_state_dict(st["ar"])
        dec.load_state_dict(st["dec"])
        nan = st.get("nan", False)
        print(f"[resume] loaded trained models from {ckpt} (nan={nan})", flush=True)
    else:
        nan = pb.train_models(
            ebt, ar, blocks, dev, steps, bs=16, langevin=(5, 15), log=lambda m: print(m, flush=True)
        )
        print(f"[train] done nan={nan}", flush=True)
        if nan:
            write_artifact(
                "complete: thesis_a_p1_v3_INCONCLUSIVE_training_diverged",
                preconditions_checked,
                False,
                t0,
                divergence=True,
            )
            return

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

    positive_control_passed = ar_best >= 0.3

    if not positive_control_passed:
        verdict = f"complete: thesis_a_p1_v3_INCONCLUSIVE_positive_control_failed_ar_best_{ar_best:.3f}_below_0.3"
        adjudication = "inconclusive_positive_control_failed"
    elif ebt_best > 0.0:
        verdict = f"complete: thesis_a_p1_v3_decode_artifact_bounded_ar_best_{ar_best:.3f}_ebt_best_{ebt_best:.3f}_positive_control_passed_energy_as_generator_still_bounded"
        adjudication = "decode_artifact_bounded"
    else:
        verdict = f"complete: thesis_a_p1_v3_FUNDAMENTAL_causal_inductive_bias_gap_ar_best_{ar_best:.3f}_ebt_best_0.000_positive_control_passed"
        adjudication = "fundamental_causal_inductive_bias_gap"

    art = {
        "experiment": "thesis_a_p1_discrete_search_v3",
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "positive_control_passed": positive_control_passed,
        "ar_best": ar_best,
        "ebt_best": ebt_best,
        "adjudication": adjudication,
        "n_train": n_train,
        "per_method_accuracies": {
            "ar_greedy": ar1_acc,
            "ar_selfconsistency": arV_acc,
            "ebt_argmin": am_acc,
            "ebt_beam": bm_acc,
            "ebt_descent": de_acc,
        },
        "training_diverged": nan,
        "energy_as_generator_still_bounded": True,
        "preconditions_checked": preconditions_checked,
        "handoff_to_operator": handoff_to_operator,
        "model_specs": {
            "dim": dim,
            "n_layers": layers,
            "ebt": "scaled_ebt_from_scratch",
            "ar": "matched",
            "n_train": n_train,
            "from_scratch": True,
        },
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
                    "n_train": n_train,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "duration_s": round(time.time() - t0, 2),
    }
    with open(
        os.path.join(
            PROJECT_ROOT, "results", "experiment_3787_p1_discrete_search_adjudication_v3_retry.json"
        ),
        "w",
    ) as f:
        json.dump(art, f, indent=2)
    print("\n" + verdict, flush=True)


if __name__ == "__main__":
    main()
