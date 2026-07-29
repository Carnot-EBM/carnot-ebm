"""GAP-3 Stage 2 v2: the panel-specified retry of the trained ARC transition-EBM — structured
same-shape near-miss curriculum + REAL mined TRM errors + a spatially-resolving energy head.

WHAT v1 GOT WRONG (results/arc3_gap3_stage2_adversarial_verify.json, 5/5 NEGATIVE_CONFIRMED, RECIPE):
  1. NEGATIVES: only 3.8% of TRM's real wrong candidates matched v1's synthetic corruption families;
     91.5% of real errors are SAME-SHAPE near-misses / plausible-but-wrong rule applications. v1's
     energy was at chance (AUROC 0.481) exactly there.
  2. ARCHITECTURE: v1's global mean-pooled CNN could not resolve 1-2-cell differences even in
     principle (c3202e5a: gold one cell from the vote leader, ranked 747/754).
THE v2 FIXES (all three panel axes):
  1. CURRICULUM: per positive, negatives = up to 3 REAL TRM wrong candidates mined from the TRAINING
     split (arc3_gap3_stage2v2_mine_real_negs.py; zero eval-task leak, hard-asserted) + structured
     same-shape near-misses (single-component recolor / move / delete, component color-swap, 1-3
     targeted cell flips) + 2 v1 coverage corruptions.
  2. ARCHITECTURE: keep the v1 relation skeleton but add a stride-1 high-resolution trunk and a
     FiLM-conditioned LOCAL ENERGY MAP: the rule embedding modulates the candidate's 30x30 feature
     map; E = global-MLP term + mean(local map) + 0.5 * max(local map). The max term makes a single
     bad cell raise the energy — the resolution v1 structurally lacked.
  3. GATES (panel): the selection eval runs ONLY if gold-vs-structured-same-shape-near-miss AUROC
     > 0.70 on held-out TRAINING tasks with FRESH negatives (v1 was at 0.481 — "predictably doomed").
     At eval: pass@2 must beat the exact random-ranker baseline (0.1432) before vote (0.4516) even
     matters. Mean-val reported alongside best-val (v1's best-val carried ~6-7pt selection optimism).
HONESTY DISCLOSURE: mining negatives from the SAME TRM checkpoint that produced the eval candidates
forfeits STRICT generator-independence (the energy is tuned to TRM's error distribution; task-split
hygiene fully preserved). The panel pre-approved this trade. The energy itself still reads ONLY
(demos, test_input, candidate_grid) at inference — no generator state (REQ-GAP3-2 intact).

  # CPU smoke: --steps 30 --eval_every 15 --batch 8 --device cpu
  # real run (conductor paused): mine first, then
  ~/trm_venv/bin/python scripts/experiments/arc3_gap3_stage2v2_transition_ebm.py --mode all
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — the universal torch convention

import sys
from carnot.paths import repo_root

# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
CARNOT = str(repo_root())
sys.path.insert(0, f"{CARNOT}/scripts/experiments")

from arc3_gap3_stage2_transition_ebm import (  # noqa: E402  (v1 substrate, reused not duplicated)
    DIHEDRAL,
    EMB,
    K_NEG,
    MAX_DEMOS,
    MAX_HW,
    N_COLORS,
    POOL,
    SEED,
    _auroc,
    _fit_logreg,
    _grouped_loto_union,
    _pass,
    apply_color_perm,
    build_examples,
    clip_grid,
    encode_pair,
    ghash,
    load_training_corpus,
    make_negatives,
    split_train_val,
)

MINED = f"{CARNOT}/results/arc3_gap3_stage2v2_mined_negs.json.gz"
CKPT = f"{CARNOT}/results/arc3_gap3_stage2v2_ebm.pt"
ARTIFACT = f"{CARNOT}/results/arc3_gap3_stage2v2_transition_ebm.json"
RANDOM_BASELINE_NOTE = "exact E[pass@2] of a uniform-random ranker on this pool (panel fix)"


# ------------------------------------------------------------------- structured near-miss negatives
def components(a):
    """4-connected same-color components of non-background cells. Returns list of (color, [(y,x)...])."""
    a = np.asarray(a)
    seen = np.zeros(a.shape, dtype=bool)
    out = []
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            if a[y, x] == 0 or seen[y, x]:
                continue
            color, stack, cells = int(a[y, x]), [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < a.shape[0]
                        and 0 <= nx < a.shape[1]
                        and not seen[ny, nx]
                        and a[ny, nx] == color
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append((color, cells))
    return out


def near_miss_negatives(rng, gold, k, max_frac=None):
    """k SAME-SHAPE structured near-misses of gold — the 91.5% class v1 never trained on. Families:
    single-component recolor / 1-cell move / delete, two-component color swap, 1-3 targeted cell flips.
    All preserve gold's shape exactly. max_frac (e.g. 0.05) additionally bounds the changed-cell
    fraction (used by the gate's <=5%-cells near-miss definition)."""
    gold = clip_grid(gold)
    comps = components(gold)
    seen = {ghash(gold)}
    negs = []

    def bounded(cand):
        if max_frac is None:
            return True
        limit = max(1, int(math.ceil(gold.size * max_frac)))
        return int((cand != gold).sum()) <= limit

    def push(cand):
        if cand is None or cand.shape != gold.shape:
            return
        h = ghash(cand)
        if h not in seen and bounded(cand):
            seen.add(h)
            negs.append(cand)

    def comp_recolor():
        if not comps:
            return None
        color, cells = comps[rng.integers(len(comps))]
        new = int(rng.choice([c for c in range(1, N_COLORS) if c != color]))
        a = gold.copy()
        for y, x in cells:
            a[y, x] = new
        return a

    def comp_move():
        if not comps:
            return None
        color, cells = comps[rng.integers(len(comps))]
        dy, dx = [(0, 1), (0, -1), (1, 0), (-1, 0)][rng.integers(4)]
        a = gold.copy()
        for y, x in cells:
            a[y, x] = 0
        for y, x in cells:
            ny, nx = y + dy, x + dx
            if 0 <= ny < a.shape[0] and 0 <= nx < a.shape[1]:
                a[ny, nx] = color
        return a

    def comp_delete():
        if not comps:
            return None
        _, cells = comps[rng.integers(len(comps))]
        a = gold.copy()
        for y, x in cells:
            a[y, x] = 0
        return a

    def comp_swap():
        distinct = list({c for c, _ in comps})
        if len(distinct) < 2:
            return None
        c1, c2 = rng.choice(distinct, 2, replace=False)
        a = gold.copy()
        a[gold == c1], a[gold == c2] = c2, c1
        return a

    def cell_flips():
        a = gold.copy()
        k_cells = 1 + int(rng.integers(3))
        nz = np.argwhere(gold > 0)
        for _ in range(k_cells):
            if (
                len(nz) and rng.random() < 0.7
            ):  # bias toward content cells — junk flips are too easy
                y, x = nz[rng.integers(len(nz))]
            else:
                y, x = rng.integers(a.shape[0]), rng.integers(a.shape[1])
            a[y, x] = int(rng.choice([c for c in range(N_COLORS) if c != a[y, x]]))
        return a

    fams = [comp_recolor, comp_move, comp_delete, comp_swap, cell_flips]
    tries = 0
    while len(negs) < k and tries < k * 12:
        tries += 1
        cand = fams[rng.integers(len(fams))]()
        if cand is not None:
            push(np.asarray(cand))
    while len(negs) < k and gold.size > 1:  # guaranteed fallback: 1-cell flips always exist
        cand = cell_flips()
        push(np.asarray(cand))
        tries += 1
        if tries > k * 30:
            break
    return negs


# ------------------------------------------------------------------------------- mined negatives
def load_mined():
    """{(task, input_hash): [neg grids]} from the miner output; {} if the file is absent (the trainer
    then runs structured-only and the artifact discloses it)."""
    if not Path(MINED).exists():
        return {}, {"available": False}
    with gzip.open(MINED, "rt") as f:
        m = json.load(f)
    lut = {}
    for task, slots in m["tasks"].items():
        for ih, s in slots.items():
            lut[(task, ih)] = [clip_grid(g) for g in s["negs"]]
    meta = {
        "available": True,
        "n_tasks_with_negs": m["n_tasks_with_negs"],
        "n_distinct_negs": m["n_distinct_negs"],
        "n_rows_scanned": m["n_rows_scanned"],
        "n_wrong_rows": m["n_wrong_rows"],
        "generator": m["generator"],
    }
    return lut, meta


# ------------------------------------------------------------------------------------- model v2
class SpatialPairEncoder(nn.Module):
    """v2 encoder: a stride-1 high-resolution trunk (the v1 killer fix — local detail survives) plus
    the v1-style downsampling path for the global embedding. Returns (hi-res map, embedding, mask)."""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(2 * (N_COLORS + 1), 48, 3, padding=1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        down = []
        ch = [64, 96, 128, 192]
        for i in range(3):
            down += [
                nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1),
                nn.GroupNorm(8, ch[i + 1]),
                nn.GELU(),
            ]
        self.down = nn.Sequential(*down)
        self.proj = nn.Linear(2 * ch[-1], EMB)
        self.norm = nn.LayerNorm(EMB)

    def forward(self, x):  # (B, 22, 30, 30)
        m = self.trunk(x)
        h = self.down(m)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=-1)
        emb = self.norm(self.proj(pooled))
        mask = (x[:, N_COLORS] + x[:, 2 * N_COLORS + 1]).clamp(max=1.0)  # input OR output in-bounds
        return m, emb, mask


class SpatialTransitionEBM(nn.Module):
    """E = global relation term + FiLM-conditioned local energy map (mean + 0.5*max over valid cells).
    The max term is the 1-bad-cell detector v1 structurally lacked."""

    def __init__(self):
        super().__init__()
        self.enc = SpatialPairEncoder()
        self.film = nn.Linear(EMB, 2 * 64)
        self.local = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(4 * EMB, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1)
        )

    def energy(self, rule_emb, cand_map, cand_emb, cand_mask):
        gamma, beta = self.film(rule_emb).chunk(2, dim=-1)
        h = cand_map * (1 + gamma[..., None, None]) + beta[..., None, None]
        emap = self.local(h).squeeze(1)  # (B, 30, 30)
        denom = cand_mask.sum(dim=(1, 2)).clamp(min=1.0)
        e_mean = (emap * cand_mask).sum(dim=(1, 2)) / denom
        e_max = (emap + (cand_mask - 1.0) * 1e4).amax(dim=(1, 2))  # -inf outside the valid mask
        z = torch.cat(
            [rule_emb, cand_emb, rule_emb * cand_emb, (rule_emb - cand_emb).abs()], dim=-1
        )
        return self.head(z).squeeze(-1) + e_mean + 0.5 * e_max


# ------------------------------------------------------------------------------------ batching v2
def augment_instance_v2(rng, demos, tin, tout):
    """Like v1's augment_instance but ALSO returns the transform so mined negatives (stored in the
    original task frame) can be mapped into the augmented frame consistently."""
    d = DIHEDRAL[rng.integers(8)]
    perm = np.arange(N_COLORS)
    perm[1:] = rng.permutation(perm[1:])

    def f(g):
        return apply_color_perm(d(clip_grid(g)).copy(), perm)

    demos2 = [{"input": f(p["input"]), "output": f(p["output"])} for p in demos]
    return demos2, f(tin), f(tout), f


def collate_v2(rng, batch_examples, gold_pool, mined_lut, augment=True):
    """[gold] + up to 3 mined real negatives + structured near-misses + 2 v1 coverage corruptions,
    deduped by hash, capped at 1+K_NEG. Mined negatives join by (task, input-hash in ORIGINAL frame)
    and are then mapped through the instance augmentation."""
    demo_t, demo_m, cand_t, n_cands, n_mined_used = [], [], [], [], 0
    for name, demos, tin, tout in batch_examples:
        ih = ghash(clip_grid(tin))
        mined_raw = mined_lut.get((name, ih), [])
        if augment:
            demos2, tin2, tout2, f = augment_instance_v2(rng, demos, tin, tout)
        else:
            demos2 = [
                {"input": clip_grid(p["input"]), "output": clip_grid(p["output"])} for p in demos
            ]
            tin2, tout2, f = clip_grid(tin), clip_grid(tout), clip_grid
        gold_h = ghash(np.asarray(tout2))
        seen = {gold_h}
        negs = []
        for g in mined_raw:
            if len(negs) >= 3:
                break
            cand = np.asarray(f(g))
            h = ghash(cand)
            if h not in seen:
                seen.add(h)
                negs.append(cand)
        n_mined_used += len(negs)
        for cand in near_miss_negatives(rng, tout2, k=K_NEG - len(negs) - 2):
            h = ghash(cand)
            if h not in seen:
                seen.add(h)
                negs.append(cand)
        for cand in make_negatives(rng, tout2, tin2, demos2, gold_pool, k=2):
            h = ghash(cand)
            if h not in seen and len(negs) < K_NEG:
                seen.add(h)
                negs.append(cand)
        if len(demos2) > MAX_DEMOS:
            demos2 = [demos2[j] for j in rng.choice(len(demos2), MAX_DEMOS, replace=False)]
        dt = torch.zeros(MAX_DEMOS, 2 * (N_COLORS + 1), MAX_HW, MAX_HW)
        dm = torch.zeros(MAX_DEMOS)
        for j, p in enumerate(demos2):
            dt[j] = encode_pair(p["input"], p["output"])
            dm[j] = 1.0
        cands = [np.asarray(tout2)] + negs
        ct = torch.stack([encode_pair(tin2, c) for c in cands])
        if len(cands) < 1 + K_NEG:
            ct = torch.cat([ct, torch.zeros(1 + K_NEG - len(cands), *ct.shape[1:])])
        demo_t.append(dt)
        demo_m.append(dm)
        cand_t.append(ct)
        n_cands.append(len(cands))
    return (
        torch.stack(demo_t),
        torch.stack(demo_m),
        torch.stack(cand_t),
        torch.tensor(n_cands),
        n_mined_used,
    )


def forward_batch_v2(model, demo_t, demo_m, cand_t, device):
    B, D = demo_t.shape[:2]
    Kc = cand_t.shape[1]
    _, de, _ = model.enc(demo_t.reshape(B * D, *demo_t.shape[2:]).to(device))
    de = de.reshape(B, D, EMB)
    m = demo_m.to(device).unsqueeze(-1)
    rule = (de * m).sum(1) / m.sum(1).clamp(min=1.0)
    cm, ce, cmask = model.enc(cand_t.reshape(B * Kc, *cand_t.shape[2:]).to(device))
    rule_x = rule.unsqueeze(1).expand(-1, Kc, -1).reshape(B * Kc, EMB)
    E = model.energy(rule_x, cm, ce, cmask).reshape(B, Kc)
    return E


# ---------------------------------------------------------------------------------------- train
def train(args, device):
    t0 = time.time()
    tasks = load_training_corpus()
    train_names, val_names = split_train_val(tasks)
    pool_tasks = set()
    with gzip.open(POOL, "rt") as f:
        for e in json.load(f)["entries"]:
            pool_tasks.add(e["task"])
    assert not (pool_tasks & set(tasks)), "SPLIT LEAK: eval-pool tasks inside training corpus"

    mined_lut, mined_meta = load_mined()
    if mined_meta["available"]:
        mined_train = {k for k in mined_lut if k[0] in set(train_names)}
        mined_val = {k for k in mined_lut if k[0] in set(val_names)}
        print(
            f"[train] mined negatives: {mined_meta['n_distinct_negs']} across "
            f"{mined_meta['n_tasks_with_negs']} tasks ({len(mined_train)} train-task slots, "
            f"{len(mined_val)} val-task slots)",
            flush=True,
        )
    else:
        print(
            "[train] WARNING: no mined-negatives file — structured-only curriculum "
            "(disclosed in artifact)",
            flush=True,
        )

    train_ex = build_examples(tasks, train_names)
    val_ex = build_examples(tasks, val_names)
    gold_pool = [clip_grid(p["output"]) for n in train_names for p in tasks[n]["train"][:2]]
    rng = np.random.default_rng(SEED)
    vrng = np.random.default_rng(SEED + 1)

    model = SpatialTransitionEBM().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, s / 300) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))
        ),
    )

    def eval_val(n_batches=12):
        model.eval()
        accs = []
        with torch.no_grad():
            for _ in range(n_batches):
                idx = vrng.choice(len(val_ex), min(args.batch, len(val_ex)), replace=False)
                bt = [val_ex[i] for i in idx]
                dt, dm, ct, nc, _ = collate_v2(vrng, bt, gold_pool, mined_lut, augment=False)
                E = forward_batch_v2(model, dt, dm, ct, device)
                for b in range(len(bt)):
                    e = E[b, : nc[b]]
                    accs.append(1.0 if int((e < e[0]).sum().item()) == 0 else 0.0)
        model.train()
        return float(np.mean(accs))

    best_val, best_state, bad, log, total_mined = -1.0, None, 0, [], 0
    model.train()
    for step in range(1, args.steps + 1):
        idx = rng.choice(len(train_ex), args.batch, replace=False)
        bt = [train_ex[i] for i in idx]
        dt, dm, ct, nc, nm = collate_v2(rng, bt, gold_pool, mined_lut, augment=True)
        total_mined += nm
        E = forward_batch_v2(model, dt, dm, ct, device)
        logits = -E
        for b in range(len(bt)):
            logits[b, nc[b] :] = -1e9
        loss = F.cross_entropy(logits, torch.zeros(len(bt), dtype=torch.long, device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0:
            vacc = eval_val()
            log.append(
                {"step": step, "loss": round(float(loss.item()), 4), "val_top1_acc": round(vacc, 4)}
            )
            print(
                f"[train] step {step}/{args.steps} loss={loss.item():.4f} val_top1={vacc:.4f} "
                f"(mined used so far: {total_mined})",
                flush=True,
            )
            if vacc > best_val:
                best_val, bad = vacc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"[train] early stop at {step}", flush=True)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    last10 = [r["val_top1_acc"] for r in log[-10:]]
    mean_val_last10 = round(float(np.mean(last10)), 4) if last10 else None
    corpus_hash = hashlib.sha256(",".join(sorted(tasks)).encode()).hexdigest()[:16]
    torch.save(
        {
            "state_dict": model.state_dict(),
            "best_val_top1": best_val,
            "mean_val_top1_last10": mean_val_last10,
            "log": log,
            "train_names": train_names,
            "val_names": val_names,
            "n_params": n_params,
            "corpus_hash": corpus_hash,
            "config": vars(args),
            "seed": SEED,
            "mined_meta": mined_meta,
            "total_mined_negs_used": total_mined,
        },
        CKPT,
    )
    dur = time.time() - t0
    print(
        f"[train] DONE in {dur:.0f}s best_val={best_val:.4f} mean_val_last10={mean_val_last10} "
        f"params={n_params}",
        flush=True,
    )
    return {
        "train_duration_s": round(dur, 1),
        "best_val_top1_acc": round(best_val, 4),
        "mean_val_top1_last10": mean_val_last10,
        "n_params": n_params,
        "n_train_tasks": len(train_names),
        "n_val_tasks": len(val_names),
        "corpus_hash": corpus_hash,
        "mined_meta": mined_meta,
        "total_mined_negs_used": total_mined,
        "train_log": log,
    }


# ------------------------------------------------------------------------------------ gate 1
def gate1_near_miss_auroc(model, device, tasks, val_names, n_examples=150, seed=SEED + 777):
    """THE PANEL GATE: gold-vs-structured-same-shape-near-miss (<=5% cells) AUROC on held-out
    TRAINING tasks, FRESH negatives (fresh seed, never seen in training). v1 measured 0.481 here
    post-hoc; the selection eval runs only if this clears 0.70."""
    rng = np.random.default_rng(seed)
    val_ex = build_examples(tasks, val_names)
    idx = rng.choice(len(val_ex), min(n_examples, len(val_ex)), replace=False)
    aurocs = []
    model.eval()
    with torch.no_grad():
        for i in idx:
            name, demos, tin, tout = val_ex[i]
            demos_c = [
                {"input": clip_grid(p["input"]), "output": clip_grid(p["output"])} for p in demos
            ][:MAX_DEMOS]
            tout_c = clip_grid(tout)
            negs = near_miss_negatives(rng, tout_c, k=10, max_frac=0.05)
            if len(negs) < 3:
                continue
            dt = torch.zeros(1, MAX_DEMOS, 2 * (N_COLORS + 1), MAX_HW, MAX_HW)
            dm = torch.zeros(1, MAX_DEMOS)
            for j, p in enumerate(demos_c):
                dt[0, j] = encode_pair(p["input"], p["output"])
                dm[0, j] = 1.0
            cands = [tout_c] + negs
            ct = torch.stack([encode_pair(clip_grid(tin), c) for c in cands]).unsqueeze(0)
            E = forward_batch_v2(model, dt, dm, ct, device)[0, : len(cands)].cpu().numpy()
            au = _auroc([-E[0]], list(-E[1:]))
            if au is not None:
                aurocs.append(au)
    return (round(float(np.mean(aurocs)), 4) if aurocs else None), len(aurocs)


# ---------------------------------------------------------------------------------------- eval
def score_pool(model, device, entries, tta=False):
    """Per-candidate energies; tta=True averages over the 8 dihedral transforms applied consistently
    to demos + test_input + candidate (the panel-demonstrated free +0.03)."""
    out = []
    transforms = DIHEDRAL if tta else DIHEDRAL[:1]
    with torch.no_grad():
        for e in entries:
            acc = np.zeros(len(e["candidates"]))
            for d in transforms:
                demos = [
                    {
                        "input": d(clip_grid(p["input"])).copy(),
                        "output": d(clip_grid(p["output"])).copy(),
                    }
                    for p in e["demos"]
                ][:MAX_DEMOS]
                tin = d(clip_grid(e["test_input"])).copy()
                dt = torch.zeros(1, MAX_DEMOS, 2 * (N_COLORS + 1), MAX_HW, MAX_HW)
                dm = torch.zeros(1, MAX_DEMOS)
                for j, p in enumerate(demos):
                    dt[0, j] = encode_pair(p["input"], p["output"])
                    dm[0, j] = 1.0
                _, de, _ = model.enc(dt[0, : len(demos)].to(device))
                rule = de.mean(0, keepdim=True)
                Es = []
                grids = [d(clip_grid(c["grid"])).copy() for c in e["candidates"]]
                for i0 in range(0, len(grids), 192):
                    ct = torch.stack([encode_pair(tin, g) for g in grids[i0 : i0 + 192]])
                    cm, ce, cmask = model.enc(ct.to(device))
                    Es.append(model.energy(rule.expand(len(ce), -1), cm, ce, cmask).cpu().numpy())
                acc += np.concatenate(Es)
            out.append(acc / len(transforms))
    return out


def evaluate_pool(args, device, train_meta, gate1, gate1_n):
    t0 = time.time()
    ck = torch.load(CKPT, map_location="cpu")
    model = SpatialTransitionEBM().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with gzip.open(POOL, "rt") as f:
        pool = json.load(f)
    entries = pool["entries"]
    _ENTRY_KEYS = {"task", "demos", "test_input", "candidates"}
    _CAND_KEYS = {"votes", "q_mean", "correct", "grid"}
    for e in entries:
        assert set(e.keys()) == _ENTRY_KEYS
        for c in e["candidates"]:
            assert set(c.keys()) == _CAND_KEYS

    E_plain = score_pool(model, device, entries, tta=False)
    E_tta = score_pool(model, device, entries, tta=True)

    tasks = []
    for e, ep, et in zip(entries, E_plain, E_tta):
        tot = sum(c["votes"] for c in e["candidates"])
        cands = [
            {
                "votes": c["votes"],
                "q_mean": c["q_mean"],
                "correct": c["correct"],
                "E": float(a),
                "E_tta": float(b),
                "vote_share": c["votes"] / max(1, tot),
            }
            for c, a, b in zip(e["candidates"], ep, et)
        ]
        tasks.append({"task": e["task"], "cands": cands})

    n = len(tasks)
    n_oracle = sum(1 for t in tasks if any(c["correct"] for c in t["cands"]))
    oracle2 = round(n_oracle / n, 4)
    # exact uniform-random pass@2 baseline (panel required): E[hits] = sum over tasks of
    # P(gold in a random top-2) = n_gold-in-top2 hypergeometric = min(2,n)/n when exactly 1 gold
    rand_terms = []
    for t in tasks:
        ng = sum(1 for c in t["cands"] if c["correct"])
        nc = len(t["cands"])
        rand_terms.append(
            0.0
            if ng == 0
            else 1.0 - math.comb(nc - ng, min(2, nc)) / math.comb(nc, min(2, nc))
            if nc >= 2
            else 1.0
        )
    random_baseline = round(float(np.mean(rand_terms)), 4)

    allc = [c for t in tasks for c in t["cands"]]
    vs = np.array([c["votes"] for c in allc], float)
    for key in ["E", "E_tta"]:
        es = np.array([c[key] for c in allc], float)
        beta = np.cov(vs, es)[0, 1] / (vs.var() or 1.0)
        alpha = es.mean() - beta * vs.mean()
        for c in allc:
            c[f"_{key}_resid"] = c[key] - (alpha + beta * c["votes"])

    _grouped_loto_union(
        tasks, lambda c: np.array([np.log1p(c["votes"]), c["vote_share"], c["q_mean"]])
    )
    for t in tasks:
        for c in t["cands"]:
            c["_union_noE"] = c["_u"]
    _grouped_loto_union(
        tasks, lambda c: np.array([np.log1p(c["votes"]), c["vote_share"], c["q_mean"], c["E_tta"]])
    )
    for t in tasks:
        for c in t["cands"]:
            c["_union_withE"] = c["_u"]

    rankers = {
        "TRM_VOTE": lambda c: (-c["votes"],),
        "EBM_v2": lambda c: (c["E"],),
        "EBM_v2_TTA8": lambda c: (c["E_tta"],),
        "EBM_v2_TTA8_residual_global": lambda c: (c["_E_tta_resid"], -c["votes"]),
        "HYBRID_vote_then_E_tta": lambda c: (-c["votes"], c["E_tta"]),
        "UNION_votes_qmean_voteshare": lambda c: (c["_union_noE"], -c["votes"]),
        "UNION_plus_E_tta": lambda c: (c["_union_withE"], -c["votes"]),
    }
    res = {name: _pass(tasks, key) for name, key in rankers.items()}

    def auroc_suite(score_fn):
        per, hard = [], []
        for t in tasks:
            g = [score_fn(c) for c in t["cands"] if c["correct"]]
            ng = [score_fn(c) for c in t["cands"] if not c["correct"]]
            au = _auroc(g, ng)
            if au is not None:
                per.append(au)
            ngh = [score_fn(c) for c in t["cands"] if not c["correct"] and c["votes"] >= 5]
            auh = _auroc(g, ngh)
            if auh is not None:
                hard.append(auh)
        return (
            round(float(np.mean(per)), 4) if per else None,
            round(float(np.mean(hard)), 4) if hard else None,
        )

    auroc_plain, hard_plain = auroc_suite(lambda c: -c["E"])
    auroc_tta, hard_tta = auroc_suite(lambda c: -c["E_tta"])

    def _lcg(seed):
        x = seed
        while True:
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            yield x

    def _boot(kA, kB):
        gen, deltas = _lcg(SEED), []

        def p2(sample, key):
            return sum(
                int(any(c["correct"] for c in sorted(t["cands"], key=key)[:2])) for t in sample
            ) / len(sample)

        for _ in range(1000):
            samp = [tasks[next(gen) % n] for _ in range(n)]
            deltas.append(p2(samp, kA) - p2(samp, kB))
        deltas.sort()
        return [round(deltas[25], 4), round(deltas[974], 4)]

    vote2 = res["TRM_VOTE"]["pass@2"]
    e2 = res["EBM_v2_TTA8"]["pass@2"]
    gates = {
        "gate1_near_miss_auroc_gt_0p70": bool((gate1 or 0) > 0.70),
        "gate1_value": gate1,
        "gate1_n_examples": gate1_n,
        "selection_beats_random": bool(e2 > random_baseline),
        "selection_beats_vote": bool(e2 > vote2),
        "union_value_add": bool(
            res["UNION_plus_E_tta"]["pass@2"] > res["UNION_votes_qmean_voteshare"]["pass@2"]
        ),
        "headroom_capture_fraction": round((e2 - vote2) / max(1e-9, oracle2 - vote2), 4),
    }
    verdict = (
        "complete: gap3_stage2v2_"
        + (
            "BEATS_vote"
            if gates["selection_beats_vote"]
            else (
                "beats_random_not_vote" if gates["selection_beats_random"] else "at_or_below_random"
            )
        )
        + f"_n{n}_vote_{vote2}_ebmtta_{e2}_rand_{random_baseline}"
        + f"_gate1_{gate1}_auroctta_{auroc_tta}_unionadd_{gates['union_value_add']}"
    )
    art = {
        "experiment": "arc3_gap3_stage2v2_transition_ebm",
        "title": "GAP-3 Stage 2 v2: mined-real + structured-near-miss curriculum, spatial FiLM energy",
        "honest_verdict": verdict,
        "inference_substrate": "live_gpu_ebm_train_plus_offline_trm_candidate_rerank_no_oracle",
        "n_tasks": n,
        "n_oracle_hit": n_oracle,
        "oracle_pass2_ceiling": oracle2,
        "random_ranker_pass2_baseline": random_baseline,
        "random_baseline_note": RANDOM_BASELINE_NOTE,
        "rankers": res,
        "auroc": {
            "ebm_macro": auroc_plain,
            "ebm_hard_neg_votes_ge5": hard_plain,
            "ebm_tta_macro": auroc_tta,
            "ebm_tta_hard_neg_votes_ge5": hard_tta,
            "vote_macro_reference": 0.9235,
        },
        "gates": gates,
        "bootstrap": {
            "ebm_tta_vs_vote_pass2_ci95": _boot(rankers["EBM_v2_TTA8"], rankers["TRM_VOTE"]),
            "union_plus_E_vs_union_pass2_ci95": _boot(
                rankers["UNION_plus_E_tta"], rankers["UNION_votes_qmean_voteshare"]
            ),
            "B": 1000,
        },
        "training": train_meta,
        "generator_independence_disclosure": (
            "FORFEITED at the negative-mining level (panel-approved trade): up to 3 negatives per "
            "positive are real wrong candidates mined from TRM arc_v1 on its own TRAINING split "
            "(zero eval-task leak, hard-asserted). The energy itself reads only (demos, test_input, "
            "candidate_grid) at inference — REQ-GAP3-2 intact."
        ),
        "model_specs": {
            "architecture": "SpatialPairEncoder (stride-1 hi-res trunk + downsample path, mean+max "
            "pool) -> 256-d; rule = mean over demo embeddings; E = global relation MLP "
            "+ FiLM-conditioned local energy map (mean + 0.5*max over valid cells)",
            "n_params": train_meta.get("n_params"),
            "curriculum": "per positive: <=3 mined real TRM errors + structured same-shape "
            "near-misses (component recolor/move/delete/swap, 1-3 cell flips) + 2 v1 "
            "coverage corruptions",
            "tta": "dihedral-8 energy averaging at eval",
        },
        "preconditions_checked": [
            {"resource": "cuda_available", "available": bool(torch.cuda.is_available())},
            {"resource": "eval_pool_export", "available": Path(POOL).exists()},
            {
                "resource": "mined_negatives",
                "available": train_meta.get("mined_meta", {}).get("available", False),
            },
        ],
        "random_seed": SEED,
        "reproducibility_checksum": train_meta.get("corpus_hash"),
        "duration_s": round(time.time() - t0 + train_meta.get("train_duration_s", 0.0), 1),
        "eval_duration_s": round(time.time() - t0, 1),
    }
    Path(ARTIFACT).write_text(json.dumps(art, indent=2, sort_keys=True) + "\n")
    print(f"-> {verdict}")
    for r in rankers:
        print(f"   {r:34s} pass@1={res[r]['pass@1']} pass@2={res[r]['pass@2']}")
    print(f"   oracle={oracle2} random={random_baseline} gate1={gate1} (n={gate1_n})")
    print(f"   auroc: plain={auroc_plain}/{hard_plain} tta={auroc_tta}/{hard_tta} (vote 0.9235)")
    print(f"   gates={gates}")
    print(
        f"   bootstrap tta-vote={art['bootstrap']['ebm_tta_vs_vote_pass2_ci95']} "
        f"union+E-union={art['bootstrap']['union_plus_E_vs_union_pass2_ci95']}"
    )
    return art


def write_gate_blocked_artifact(train_meta, gate1, gate1_n):
    """Panel rule: 'do not spend another selection run below that bar.' The gate failure IS the
    result — an honest negative at the cheapest possible measurement point."""
    art = {
        "experiment": "arc3_gap3_stage2v2_transition_ebm",
        "title": "GAP-3 Stage 2 v2: mined-real + structured-near-miss curriculum, spatial FiLM energy",
        "honest_verdict": (
            f"complete: gap3_stage2v2_gate1_failed_near_miss_auroc_{gate1}"
            f"_lt_0p70_no_selection_eval_run"
        ),
        "inference_substrate": "live_gpu_ebm_train_plus_offline_trm_candidate_rerank_no_oracle",
        "gates": {
            "gate1_near_miss_auroc_gt_0p70": False,
            "gate1_value": gate1,
            "gate1_n_examples": gate1_n,
            "note": "selection eval deliberately NOT run per the Stage-2 panel rule",
        },
        "training": train_meta,
        "random_seed": SEED,
        "reproducibility_checksum": train_meta.get("corpus_hash"),
        "duration_s": train_meta.get("train_duration_s", 0.0),
    }
    Path(ARTIFACT).write_text(json.dumps(art, indent=2, sort_keys=True) + "\n")
    print(f"-> {art['honest_verdict']}")
    return art


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--skip_gate", action="store_true", help="CPU smoke only — never use on a real run"
    )
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device)
    train_meta = {}
    if args.mode in ("train", "all"):
        train_meta = train(args, device)
    if args.mode in ("eval", "all"):
        ck = torch.load(CKPT, map_location="cpu")
        if not train_meta:
            train_meta = {
                "best_val_top1_acc": ck.get("best_val_top1"),
                "mean_val_top1_last10": ck.get("mean_val_top1_last10"),
                "n_params": ck.get("n_params"),
                "corpus_hash": ck.get("corpus_hash"),
                "mined_meta": ck.get("mined_meta", {}),
                "train_duration_s": 0.0,
                "reused_checkpoint": True,
            }
        model = SpatialTransitionEBM().to(device)
        model.load_state_dict(ck["state_dict"])
        tasks = load_training_corpus()
        gate1, gate1_n = (
            (None, 0)
            if args.skip_gate
            else gate1_near_miss_auroc(model, device, tasks, ck["val_names"])
        )
        print(
            f"[gate1] gold-vs-same-shape-near-miss AUROC = {gate1} over {gate1_n} examples "
            f"(bar: >0.70; v1 measured 0.481 post-hoc)",
            flush=True,
        )
        if args.skip_gate or (gate1 or 0) > 0.70:
            evaluate_pool(args, device, train_meta, gate1, gate1_n)
        else:
            write_gate_blocked_artifact(train_meta, gate1, gate1_n)


if __name__ == "__main__":
    main()
