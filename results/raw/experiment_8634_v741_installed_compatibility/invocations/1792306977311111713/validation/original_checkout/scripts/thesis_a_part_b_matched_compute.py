"""
Thesis-A part (b) — does energy-as-GENERATOR beat autoregression at MATCHED COMPUTE?

Operator-authorized 2026-06-03 direct run (task regime: tunable arithmetic; the
operator deferred to the recommended regime). Design:
docs/research-notes/phase3-thesis-a-part-b-design.md.

THE QUESTION: at EQUAL inference compute, does EBT generation beat AR generation
on held-out arithmetic accuracy?

TASK (AR-hostile by construction): fixed-width D-digit addition, answer MSD-first
("12+34=046"). MSD-first makes it AR-hostile — AR must commit the most-significant
answer digit before resolving carries from the less-significant digits (the
classic left-to-right weakness). A global energy model has a plausible edge.

EBT GENERATION (faithful energy minimisation, NOT selection among AR samples — the
disproven thesis): at each answer position evaluate the EBT's per-position energy
for EVERY candidate next token (batched) and pick the argmin-energy token. The EBT
proposes the tokens itself. Cost = VOCAB energy-evals per token.

MATCHED COMPUTE (the P0.1 lesson — matched compute, NOT matched params): give AR
self-consistency with N=VOCAB samples so AR forward passes per token == EBT energy
evals per token. Compare EBT-argmin vs AR@N at equal forward count. AR@1 greedy is
the headroom/positive-control baseline.

POSITIVE CONTROL (FALSE_NEGATIVE_RISK guard): oracle=100% by construction; the task
is tuned so AR@1 lands in a MEASURABLE, NON-SATURATED range (0.05<acc<0.95). If
AR@1 ~0 (too hard) or ~1 (too easy) the corpus has no usable headroom and any "EBT
doesn't win" reading is REJECTED, not a verdict.

KILL-GATE: PASS = EBT > AR@N at equal compute on a headroom corpus; BOUNDED = EBT
<= AR@N on a headroom corpus (honest negative); REJECTED = no headroom.

Reuses the conductor's real EBT/AR model code (experiment_3734.build_tiny_models)
+ the vendored EBTDefault, so this is a faithful test of the actual harness.
"""

import os
import sys
import time
import json
import math
import random
import hashlib
import argparse
from collections import Counter
from pathlib import Path

# Resolved from this file rather than hardcoded so a fresh clone or a
# worktree writes into ITS OWN tree. Inlined (not carnot.paths.repo_root)
# because the next line is what makes ``carnot`` importable -- importing
# the resolver here would be circular. Same rule, same answer.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "exp3734",
    os.path.join(
        PROJECT_ROOT, "scripts", "experiment_3734_fix_harness_and_bounded_train_chunk1.py"
    ),
)
exp3734 = _ilu.module_from_spec(_spec)
sys.modules["exp3734"] = exp3734
_spec.loader.exec_module(exp3734)
build_tiny_models = exp3734.build_tiny_models
EOS = exp3734.BYTE_TOKENIZER_EOS_ID
OFF = exp3734.BYTE_TOKENIZER_BYTE_OFFSET
VOCAB = exp3734.BYTE_TOKENIZER_VOCAB_SIZE

import torch
import torch.nn.functional as F  # noqa: N812 (PyTorch community convention)
import numpy as np


def enc(s):
    return [b + OFF for b in s.encode("utf-8")]


def dec_ids(ids):
    return bytes([max(0, i - OFF) for i in ids if i >= OFF]).decode("utf-8", "replace")


def make_problem(D, rng):
    a = rng.randint(0, 10**D - 1)
    b = rng.randint(0, 10**D - 1)
    return f"{a:0{D}d}+{b:0{D}d}=", f"{a + b:0{D + 1}d}"


def build_corpus(D, n, seed, exclude=None):
    """Sample up to n unique problems whose prompt is not in `exclude`. Caps the
    target to the number of available unique (a,b) pairs (10^D x 10^D) so small D
    can't infinite-loop, and bounds attempts as a hard backstop."""
    rng = random.Random(seed)
    exclude = exclude or set()
    seen, items = set(), []
    target = min(n, (10**D) ** 2 - len(exclude))
    attempts, cap = 0, max(200000, target * 200)
    while len(items) < target and attempts < cap:
        p, a = make_problem(D, rng)
        attempts += 1
        if p in exclude or p in seen:
            continue
        seen.add(p)
        items.append((p, a))
    return items


def corpus_to_blocks(items, block_size):
    stream = []
    for p, a in items:
        stream.extend(enc(p + a))
        stream.append(EOS)
    L = block_size + 1
    nb = len(stream) // L
    arr = np.asarray(stream[: nb * L], dtype=np.int64).reshape(nb, L)
    return torch.as_tensor(arr, dtype=torch.long)


def train_models(ebt, ar, blocks, device, steps, lr=3e-4, bs=16, langevin=(5, 15), log=print):
    ebt_opt = torch.optim.AdamW(ebt.parameters(), lr=lr)
    ar_opt = torch.optim.AdamW(ar.parameters(), lr=lr)
    ebt.train()
    ar.train()
    replay, buf = [], 1000
    nan = False
    for step in range(steps):
        idx = torch.randint(0, max(1, blocks.shape[0] - bs), (1,)).item()
        batch = blocks[idx : idx + bs].to(device)
        ar_opt.zero_grad(set_to_none=True)
        logits = ar(batch[:, :-1])
        ar_loss = F.cross_entropy(logits.reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
        if not torch.isfinite(ar_loss):
            nan = True
            break
        ar_loss.backward()
        torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
        ar_opt.step()
        ebt_opt.zero_grad(set_to_none=True)
        orig = ebt.token_embedding(batch[:, :-1])
        pos = ebt.token_embedding(batch[:, 1:])
        pe = ebt(orig, pos).mean()
        if replay and random.random() < 0.95:
            ib = torch.randint(0, len(replay), (bs,))
            neg = torch.stack([replay[i] for i in ib]).to(device)
        else:
            neg = torch.randn_like(pos) * 0.02
        neg = neg.detach()
        neg.requires_grad_(True)
        alpha = random.uniform(0.1, 1.0)
        for _ in range(random.randint(*langevin)):
            ne = ebt(orig.detach(), neg).mean()
            g = torch.autograd.grad(ne, neg)[0]
            neg = (neg - alpha * g + torch.randn_like(neg) * math.sqrt(2 * alpha)).detach()
            neg.requires_grad_(True)
        nef = ebt(orig, neg.detach()).mean()
        ebt_loss = pe - nef + 0.1 * (pe**2 + nef**2)
        if not torch.isfinite(ebt_loss):
            nan = True
            break
        ebt_loss.backward()
        torch.nn.utils.clip_grad_norm_(ebt.parameters(), 1.0)
        ebt_opt.step()
        for i in range(bs):
            if len(replay) < buf:
                replay.append(neg[i].detach().cpu())
            else:
                replay[random.randint(0, buf - 1)] = neg[i].detach().cpu()
        if (step + 1) % 200 == 0:
            log(
                f"[train] step={step + 1} ebt_loss={ebt_loss.item():.3f} ar_loss={ar_loss.item():.4f}"
            )
    return nan


@torch.no_grad()
def ar_greedy(ar, pid, ans_len, device):
    ids = list(pid)
    nf = 0
    for _ in range(ans_len):
        logits = ar(torch.tensor([ids], device=device))[0, -1]
        nf += 1
        ids.append(int(logits.argmax()))
    return ids[len(pid) :], nf


@torch.no_grad()
def ar_selfconsistency(ar, pid, ans_len, device, N, temp=0.8):
    """Sample N completions in PARALLEL (batch=N), majority-vote. forward count =
    N*ans_len (matched to EBT's VOCAB*ans_len when N=VOCAB)."""
    seqs = torch.tensor([pid], device=device).expand(N, -1).clone()
    for _ in range(ans_len):
        logits = ar(seqs)[:, -1]
        p = F.softmax(logits / temp, dim=-1)
        nxt = torch.multinomial(p, 1)
        seqs = torch.cat([seqs, nxt], dim=1)
    outs = [tuple(s[len(pid) :].tolist()) for s in seqs]
    maj = Counter(outs).most_common(1)[0][0]
    return list(maj), N * ans_len


@torch.no_grad()
def ebt_generate(ebt, pid, ans_len, device):
    """Faithful EBT generation by energy-argmin over the full vocab. Per position,
    batch all VOCAB candidate next tokens and pick the argmin-energy one. Cost =
    VOCAB energy-evals per token. Cannot collapse like descend-then-nearest did."""
    emb = ebt.token_embedding.weight
    cand_ids = torch.arange(VOCAB, device=device)
    cand_emb = emb[cand_ids]  # [V, dim]
    ids = list(pid)
    nf = 0
    for _ in range(ans_len):
        ctx = torch.tensor([ids], device=device)
        m = ctx.shape[1]
        orig = ebt.token_embedding(ctx).expand(VOCAB, -1, -1)  # [V, m, dim]
        if m >= 2:
            known = ebt.token_embedding(ctx[:, 1:]).expand(VOCAB, -1, -1)
        else:
            known = torch.zeros((VOCAB, 0, emb.shape[1]), device=device)
        pred = torch.cat([known, cand_emb.unsqueeze(1)], dim=1)  # [V, m, dim]
        e = ebt(orig, pred)[:, -1, 0]  # [V]
        ids.append(int(cand_ids[int(e.argmin())]))
        nf += VOCAB
    return ids[len(pid) :], nf


def eval_methods(ebt, ar, eval_items, device, D, n_eval, log=print):
    ans_len = D + 1
    N = VOCAB  # matched compute: AR samples == EBT candidate evals/token
    ar1 = arN = ebtc = 0
    ar1_nf = arN_nf = ebt_nf = 0
    samples = []
    ebt.eval()
    ar.eval()
    items = eval_items[:n_eval]
    for j, (p, true_ans) in enumerate(items):
        pid = enc(p)
        true = enc(true_ans)
        g1, nf = ar_greedy(ar, pid, ans_len, device)
        ar1_nf += nf
        ar1 += g1 == true
        gN, nf = ar_selfconsistency(ar, pid, ans_len, device, N)
        arN_nf += nf
        arN += gN == true
        ge, nf = ebt_generate(ebt, pid, ans_len, device)
        ebt_nf += nf
        ebtc += ge == true
        if j < 8:
            samples.append({"prompt": p, "true": true_ans, "ar1": dec_ids(g1), "ebt": dec_ids(ge)})
        if (j + 1) % 20 == 0:
            log(
                f"[eval] {j + 1}/{len(items)} ar1={ar1 / (j + 1):.3f} arN={arN / (j + 1):.3f} ebt={ebtc / (j + 1):.3f}"
            )
    n = len(items)
    return {
        "n_eval": n,
        "ar1_acc": ar1 / n,
        "arN_acc": arN / n,
        "ebt_acc": ebtc / n,
        "ar1_forward": ar1_nf,
        "arN_forward": arN_nf,
        "ebt_forward": ebt_nf,
        "compute_match_ratio": (ebt_nf / arN_nf) if arN_nf else None,
        "samples": samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--digits", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-eval", type=int, default=80)
    ap.add_argument("--seed", type=int, default=30603)
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.n_eval = 1000, 25

    t0 = time.time()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = torch.device(
        "cuda:1"
        if torch.cuda.device_count() > 1
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    bs_blk = 32  # arithmetic lines are ~9 chars; tight blocks = ~4x less wasted compute vs 128
    print(
        f"[setup] device={device} digits={a.digits} steps={a.steps} n_eval={a.n_eval} smoke={a.smoke}",
        flush=True,
    )

    max_unique = (10**a.digits) ** 2
    n_train = min(20000, int(max_unique * 0.7))  # leave room for a disjoint eval split
    train_items = build_corpus(a.digits, n_train, a.seed)
    tp = {t[0] for t in train_items}
    eval_items = build_corpus(a.digits, 3000, a.seed + 777, exclude=tp)  # held-out, disjoint
    blocks = corpus_to_blocks(train_items, bs_blk)
    print(
        f"[data] train={len(train_items)} eval={len(eval_items)} blocks={tuple(blocks.shape)}",
        flush=True,
    )

    ebt, ar = build_tiny_models(batch_size=16, block_size=bs_blk)
    ebt, ar = ebt.to(device), ar.to(device)
    nan = train_models(
        ebt,
        ar,
        blocks,
        device,
        a.steps,
        bs=16,
        langevin=(5, 15),
        log=lambda m: print(m, flush=True),
    )

    res = eval_methods(
        ebt, ar, eval_items, device, a.digits, a.n_eval, log=lambda m: print(m, flush=True)
    )
    ar1, arN, ebt_acc = res["ar1_acc"], res["arN_acc"], res["ebt_acc"]
    headroom = 0.05 < ar1 < 0.95
    if not headroom:
        verdict = (
            f"complete: thesis_a_part_b_REJECTED_no_headroom_ar1_{ar1:.3f}_"
            f"{'too_hard' if ar1 <= 0.05 else 'saturated'}_retune"
        )
    elif nan:
        verdict = "complete: thesis_a_part_b_INCONCLUSIVE_training_diverged"
    elif ebt_acc > arN:
        verdict = (
            f"complete: thesis_a_part_b_PASS_ebt_beats_ar_at_matched_compute_"
            f"ebt_{ebt_acc:.3f}_vs_arN_{arN:.3f}_headroom_ok"
        )
    else:
        verdict = (
            f"complete: thesis_a_part_b_BOUNDED_ebt_does_not_beat_ar_at_matched_compute_"
            f"ebt_{ebt_acc:.3f}_vs_arN_{arN:.3f}_ar1_{ar1:.3f}_headroom_ok_honest_negative"
        )

    payload = {"seed": a.seed, "digits": a.digits, "steps": a.steps, "smoke": a.smoke}
    art = {
        "experiment": "thesis_a_part_b_matched_compute",
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "operator_authorized": "2026-06-03 direct run; arithmetic regime (operator deferred to recommendation)",
        "task": f"fixed-width {a.digits}-digit addition, MSD-first answer (AR-hostile)",
        "smoke": a.smoke,
        "training_diverged": nan,
        "headroom_ok": headroom,
        "positive_control": "AR@1 in (0.05,0.95) => measurable headroom; oracle=1.0. Required before any BOUNDED verdict.",
        "ar1_greedy_acc": ar1,
        "arN_selfconsistency_acc": arN,
        "ebt_energy_argmin_acc": ebt_acc,
        "delta_ebt_minus_arN": ebt_acc - arN,
        "ebt_generation": "energy-argmin over full vocab (faithful; not selection among AR samples)",
        "matched_compute": {
            "ebt_energy_evals": res["ebt_forward"],
            "arN_forward_passes": res["arN_forward"],
            "ratio_ebt_over_ar": res["compute_match_ratio"],
            "ar_self_consistency_N": VOCAB,
        },
        "ar1_forward_passes": res["ar1_forward"],
        "n_eval": res["n_eval"],
        "ebt_sample_outputs": res["samples"],
        "model_specs": {
            "ebt": "tiny_ebt_from_scratch_byte",
            "ar": "tiny_ar_from_scratch_matched_byte",
            "ebt_decoder": "energy_argmin_over_vocab",
            "from_scratch": True,
        },
        "device": str(device),
        "random_seed": a.seed,
        "reproducibility_checksum": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest(),
        "duration_s": round(time.time() - t0, 2),
    }
    out = os.path.join(
        PROJECT_ROOT,
        "results",
        f"thesis_a_part_b_matched_compute{'_smoke' if a.smoke else ''}.json",
    )
    with open(out, "w") as f:
        json.dump(art, f, indent=2)
    print("\n" + verdict, flush=True)
    print(
        f"[done] ar1={ar1:.3f} arN={arN:.3f} ebt={ebt_acc:.3f} headroom={headroom} "
        f"ratio={res['compute_match_ratio']} dur={art['duration_s']}s -> {out}",
        flush=True,
    )
    print("[samples] " + json.dumps(res["samples"][:5]), flush=True)


if __name__ == "__main__":
    main()
