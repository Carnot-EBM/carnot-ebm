"""GAP-3 Stage 2: trained generator-INDEPENDENT ARC transition-EBM as a selection energy.

THE QUESTION. Stages 0-1 established that TRM's own model-native signals (scalar q_halt, full 512-d
penultimate latent) are partial VOTE SHADOWS — they carry no incremental gold signal beyond frequency
vote (adversarially confirmed, results/arc3_gap3_stage{0,1}_adversarial_verify.json). Stage 2 asks the
generator-independent question: can a from-scratch energy E(candidate | test_input, demo_pairs), trained
ONLY on the public ARC training corpus (never on the eval tasks, never on TRM anything), rank TRM's
correct-but-mis-voted candidates into top-2 better than TRM's own vote?

DESIGN CONSTRAINTS (from the Stage-1 adversarial round, corrigendum_2026_06_09):
  * Energy from grid CONTENT only — no generator activations, NO augmentation pooling (the structural
    vote leak that sank Stage 1: mean-pooling latents over a candidate's augmentation views baked vote
    count into the feature). Here each candidate is scored ONCE from its de-augmented grid.
  * Split hygiene: train on arc-agi_training (400 tasks) + arc-agi_concept (160). arc-agi_training2
    (ARC-AGI-2 train) is EXCLUDED — audit found 376 of ARC-1's 400 eval tasks inside it (including 29
    of our 30 eval-pool tasks): a direct task leak. The chosen corpus is exactly TRM arc_v1's own
    training distribution (arc1concept), so generator and verifier saw the same task universe and the
    eval split is held out from BOTH.
  * Baseline-to-beat for deployment value-add: the no-latent votes+q_mean+vote_share union (grouped-LOTO
    logistic, pass@2 0.4839 at Stage 1) — not just bare vote 0.4516.
  * Grouped folds: the f3e62deb task contributes two eval entries; any LOTO at eval groups them.

MODEL. A small relation-network EBM (~1.1M params): a shared CNN encodes each (input, output) grid pair
into a 256-d embedding; the task RULE embedding r = mean over demo-pair embeddings; the candidate pair
(test_input, candidate) embeds to p; energy = MLP([r, p, r*p, |r-p|]) -> scalar (LOWER = more likely the
correct rule application). Trained with InfoNCE: per (task-instance, target-pair), the gold output
competes against K generator-independent corrupted negatives (identity-copy, demo-output-copy, dihedral,
color-perm, cell-noise, row/col resize, shift, other-task output). Task-level augmentation (dihedral-8 x
color-permutation, consistent across all grids of an instance) multiplies the effective corpus.

NO-ORACLE INVARIANT (REQ-GAP3-2). At eval the energy reads (demos, test_input, candidate_grid) ONLY.
The eval pool file (results/arc3_gap3_stage2_eval_pool.json.gz) does not even contain gold output grids;
`correct` labels are used solely to SCORE rankings after the fact.

GATES (design doc Section 3 + Stage-1 corrigendum): selection pass@2(E) > pass@2(vote); within-task
AUROC > 0.70 macro AND reported on hard negatives (votes>=5) where Stage 1's gate inflated; coverage
>= 0.80 (dense energy => 1.0 by construction); headroom-capture >= 0.30; bootstrap CI over tasks.

  # train + eval (GPU; pause the conductor first — its preflight reaps non-conductor GPU procs):
  ~/trm_venv/bin/python scripts/experiments/arc3_gap3_stage2_transition_ebm.py --mode all
  # eval only (reuses the saved checkpoint):
  ~/trm_venv/bin/python scripts/experiments/arc3_gap3_stage2_transition_ebm.py --mode eval
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
from carnot.paths import repo_root

# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
CARNOT = str(repo_root())
KAGGLE = "/home/ianblenke/trm_src/kaggle/combined"
POOL = f"{CARNOT}/results/arc3_gap3_stage2_eval_pool.json.gz"
CKPT = f"{CARNOT}/results/arc3_gap3_stage2_ebm.pt"
ARTIFACT = f"{CARNOT}/results/arc3_gap3_stage2_transition_ebm.json"

SEED = 12345
MAX_HW = 30
N_COLORS = 10
EMB = 256
MAX_DEMOS = 4
K_NEG = 10


# ----------------------------------------------------------------------------- grid utilities
def ghash(g) -> str:
    """Stable content hash of a grid (list-of-lists of ints)."""
    a = np.asarray(g, dtype=np.int8)
    return hashlib.sha1(a.shape.__repr__().encode() + a.tobytes()).hexdigest()


def clip_grid(g):
    """Clamp to the legal ARC envelope: values 0-9, size <=30x30 (defensive; TRM crops already)."""
    a = np.asarray(g, dtype=np.int64)
    if a.ndim != 2:
        a = np.atleast_2d(a)
    return np.clip(a[:MAX_HW, :MAX_HW], 0, N_COLORS - 1)


DIHEDRAL = [
    lambda a: a,
    lambda a: np.rot90(a, 1),
    lambda a: np.rot90(a, 2),
    lambda a: np.rot90(a, 3),
    lambda a: np.fliplr(a),
    lambda a: np.flipud(a),
    lambda a: np.rot90(np.fliplr(a), 1),
    lambda a: np.rot90(np.fliplr(a), 3),
]


def apply_color_perm(a, perm):
    """perm is a length-10 int array; perm[0]==0 always (background is never remapped)."""
    return perm[a]


# ----------------------------------------------------------------------------- corpus
def load_training_corpus():
    """ARC-1 training (400) + ConceptARC (160). training2 EXCLUDED (376-task leak into ARC-1 eval —
    see module docstring). Returns {name: {train: [...], test: [...]}} with test outputs attached."""
    tasks = {}
    for subset in ["training", "concept"]:
        ch = json.load(open(f"{KAGGLE}/arc-agi_{subset}_challenges.json"))
        so = json.load(open(f"{KAGGLE}/arc-agi_{subset}_solutions.json"))
        for name, t in ch.items():
            test = []
            for i, pair in enumerate(t["test"]):
                out = so[name][i] if name in so and i < len(so[name]) else None
                if out is not None:
                    test.append({"input": pair["input"], "output": out})
            tasks[name] = {"train": t["train"], "test": test}
    return tasks


def split_train_val(tasks, val_frac=0.10, seed=SEED):
    rng = np.random.default_rng(seed)
    names = sorted(tasks)
    rng.shuffle(names)
    n_val = int(len(names) * val_frac)
    return names[n_val:], names[:n_val]


# ----------------------------------------------------------------------------- negatives
def make_negatives(rng, gold, test_input, demos, other_golds, k=K_NEG):
    """k generator-independent corruptions, each hash-distinct from gold (and from each other).
    Families mirror real wrong-candidate modes: identity-copy (TRM's classic failure), demo-output copy,
    wrong-orientation (dihedral), wrong-palette (color perm), near-miss cell noise, off-by-one size,
    translation, and structurally-plausible other-task outputs."""
    gold = np.asarray(gold)
    gh = ghash(gold)
    seen = {gh}
    negs = []

    def push(cand):
        if cand is None or cand.size == 0:
            return
        h = ghash(cand)
        if h not in seen:
            seen.add(h)
            negs.append(cand)

    fams = []
    fams.append(lambda: np.asarray(test_input))  # identity
    if demos:
        fams.append(lambda: np.asarray(demos[rng.integers(len(demos))]["output"]))  # demo copy
    fams.append(lambda: DIHEDRAL[1 + rng.integers(7)](gold).copy())  # orientation

    def _cperm():
        perm = np.arange(N_COLORS)
        perm[1:] = rng.permutation(perm[1:])
        return apply_color_perm(gold, perm)

    fams.append(_cperm)  # palette

    def _noise():
        a = gold.copy()
        n = max(1, int(a.size * rng.uniform(0.02, 0.30)))
        idx = rng.choice(a.size, size=min(n, a.size), replace=False)
        a.flat[idx] = rng.integers(0, N_COLORS, size=len(idx))
        return a

    fams.append(_noise)  # near-miss

    def _resize():
        a = gold
        if rng.random() < 0.5 and a.shape[0] > 1:  # delete a row/col
            ax = rng.integers(2)
            i = rng.integers(a.shape[ax])
            return np.delete(a, i, axis=ax)
        ax = rng.integers(2)
        i = int(rng.integers(a.shape[ax]))
        return np.insert(a, i, np.take(a, i, axis=ax), axis=ax)[:MAX_HW, :MAX_HW]

    fams.append(_resize)  # off-by-one size

    def _shift():
        a = gold.copy()
        dy, dx = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
        if dy == 0 and dx == 0:
            dy = 1
        out = np.zeros_like(a)
        ys, xs = (
            slice(max(0, dy), a.shape[0] + min(0, dy)),
            slice(max(0, dx), a.shape[1] + min(0, dx)),
        )
        ys2 = slice(max(0, -dy), a.shape[0] + min(0, -dy))
        xs2 = slice(max(0, -dx), a.shape[1] + min(0, -dx))
        out[ys, xs] = a[ys2, xs2]
        return out

    fams.append(_shift)  # translation
    if other_golds:
        fams.append(lambda: np.asarray(other_golds[rng.integers(len(other_golds))]))  # other-task

    tries = 0
    while len(negs) < k and tries < k * 8:
        tries += 1
        push(clip_grid(fams[rng.integers(len(fams))]()))
    return negs


# ----------------------------------------------------------------------------- encoding
def encode_pair(inp, out) -> torch.Tensor:
    """(22, 30, 30) float: per grid 10 one-hot color planes + 1 in-bounds mask, zero-padded."""
    t = torch.zeros(2 * (N_COLORS + 1), MAX_HW, MAX_HW)
    for k, g in enumerate([inp, out]):
        a = clip_grid(g)
        h, w = a.shape
        base = k * (N_COLORS + 1)
        oh = F.one_hot(torch.from_numpy(a.copy()), N_COLORS).permute(2, 0, 1).float()
        t[base : base + N_COLORS, :h, :w] = oh
        t[base + N_COLORS, :h, :w] = 1.0
    return t


class PairEncoder(nn.Module):
    """Shared CNN: a (input, output) grid pair -> 256-d embedding."""

    def __init__(self):
        super().__init__()
        ch = [2 * (N_COLORS + 1), 64, 96, 128, 192]
        blocks = []
        for i in range(4):
            blocks += [
                nn.Conv2d(ch[i], ch[i + 1], 3, stride=1 if i == 0 else 2, padding=1),
                nn.GroupNorm(8, ch[i + 1]),
                nn.GELU(),
            ]
        self.conv = nn.Sequential(*blocks)
        self.proj = nn.Linear(ch[-1], EMB)
        self.norm = nn.LayerNorm(EMB)

    def forward(self, x):  # (B, 22, 30, 30)
        h = self.conv(x).mean(dim=(2, 3))
        return self.norm(self.proj(h))


class TransitionEBM(nn.Module):
    """E(candidate | test_input, demos): rule embedding (mean over demo pairs) related to the candidate
    pair embedding through an MLP head. Lower energy = more consistent with the demonstrated rule."""

    def __init__(self):
        super().__init__()
        self.enc = PairEncoder()
        self.head = nn.Sequential(
            nn.Linear(4 * EMB, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1)
        )

    def energy(self, rule_emb, pair_emb):
        z = torch.cat(
            [rule_emb, pair_emb, rule_emb * pair_emb, (rule_emb - pair_emb).abs()], dim=-1
        )
        return self.head(z).squeeze(-1)


# ----------------------------------------------------------------------------- batching
def build_examples(tasks, names):
    """Flatten tasks into (name, demo_pairs, target_input, target_output) examples. For a target train
    pair the demos are the OTHER train pairs (within-task leave-one-out); for a test pair, all train
    pairs. Tasks contribute every pair as a target once per epoch pass."""
    examples = []
    for name in names:
        t = tasks[name]
        tr = t["train"]
        for i, p in enumerate(tr):
            demos = tr[:i] + tr[i + 1 :]
            if demos:
                examples.append((name, demos, p["input"], p["output"]))
        for p in t["test"]:
            if tr:
                examples.append((name, tr, p["input"], p["output"]))
    return examples


def augment_instance(rng, demos, tin, tout):
    """Task-consistent augmentation: ONE dihedral + ONE color permutation applied to every grid."""
    d = DIHEDRAL[rng.integers(8)]
    perm = np.arange(N_COLORS)
    perm[1:] = rng.permutation(perm[1:])
    f = lambda g: apply_color_perm(d(clip_grid(g)).copy(), perm)
    demos2 = [{"input": f(p["input"]), "output": f(p["output"])} for p in demos]
    return demos2, f(tin), f(tout)


def collate(rng, batch_examples, gold_pool, augment=True):
    """-> demo tensor (B, MAX_DEMOS, 22,30,30) + demo mask, target pair stack (B, 1+K, 22,30,30),
    n_cands per row. Negatives sampled fresh each call."""
    demo_t, demo_m, cand_t, n_cands = [], [], [], []
    for name, demos, tin, tout in batch_examples:
        if augment:
            demos, tin, tout = augment_instance(rng, demos, tin, tout)
        else:
            demos = [
                {"input": clip_grid(p["input"]), "output": clip_grid(p["output"])} for p in demos
            ]
            tin, tout = clip_grid(tin), clip_grid(tout)
        if len(demos) > MAX_DEMOS:
            demos = [demos[j] for j in rng.choice(len(demos), MAX_DEMOS, replace=False)]
        dt = torch.zeros(MAX_DEMOS, 2 * (N_COLORS + 1), MAX_HW, MAX_HW)
        dm = torch.zeros(MAX_DEMOS)
        for j, p in enumerate(demos):
            dt[j] = encode_pair(p["input"], p["output"])
            dm[j] = 1.0
        negs = make_negatives(rng, tout, tin, demos, gold_pool, k=K_NEG)
        cands = [tout] + negs
        ct = torch.stack([encode_pair(tin, c) for c in cands])
        if len(cands) < 1 + K_NEG:  # pad (masked out by n_cands)
            pad = torch.zeros(1 + K_NEG - len(cands), *ct.shape[1:])
            ct = torch.cat([ct, pad])
        demo_t.append(dt)
        demo_m.append(dm)
        cand_t.append(ct)
        n_cands.append(len(cands))
    return (
        torch.stack(demo_t),
        torch.stack(demo_m),
        torch.stack(cand_t),
        torch.tensor(n_cands),
    )


def forward_batch(model, demo_t, demo_m, cand_t, device):
    B, D = demo_t.shape[:2]
    Kc = cand_t.shape[1]
    de = model.enc(demo_t.reshape(B * D, *demo_t.shape[2:]).to(device)).reshape(B, D, EMB)
    m = demo_m.to(device).unsqueeze(-1)
    rule = (de * m).sum(1) / m.sum(1).clamp(min=1.0)
    ce = model.enc(cand_t.reshape(B * Kc, *cand_t.shape[2:]).to(device)).reshape(B, Kc, EMB)
    E = model.energy(rule.unsqueeze(1).expand(-1, Kc, -1), ce)  # (B, Kc)
    return E


# ----------------------------------------------------------------------------- training
def train(args, device):
    t0 = time.time()
    tasks = load_training_corpus()
    train_names, val_names = split_train_val(tasks)
    # split-hygiene assertion: the eval pool tasks must not be in the training corpus at all
    pool_tasks = set()
    with gzip.open(POOL, "rt") as f:
        pool = json.load(f)
    for e in pool["entries"]:
        pool_tasks.add(e["task"])
    leak = pool_tasks & set(tasks)
    assert not leak, f"SPLIT LEAK: eval-pool tasks inside training corpus: {leak}"

    train_ex = build_examples(tasks, train_names)
    val_ex = build_examples(tasks, val_names)
    gold_pool = [clip_grid(p["output"]) for n in train_names for p in tasks[n]["train"][:2]]
    rng = np.random.default_rng(SEED)
    vrng = np.random.default_rng(SEED + 1)

    model = TransitionEBM().to(device)
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
        accs, ranks = [], []
        with torch.no_grad():
            for _ in range(n_batches):
                idx = vrng.choice(len(val_ex), args.batch, replace=False)
                bt = [val_ex[i] for i in idx]
                dt, dm, ct, nc = collate(vrng, bt, gold_pool, augment=False)
                E = forward_batch(model, dt, dm, ct, device)
                for b in range(len(bt)):
                    e = E[b, : nc[b]]
                    r = int((e < e[0]).sum().item()) + 1
                    ranks.append(r)
                    accs.append(1.0 if r == 1 else 0.0)
        model.train()
        return float(np.mean(accs)), float(np.mean(ranks))

    best_val, best_state, bad = -1.0, None, 0
    log = []
    model.train()
    for step in range(1, args.steps + 1):
        idx = rng.choice(len(train_ex), args.batch, replace=False)
        bt = [train_ex[i] for i in idx]
        dt, dm, ct, nc = collate(rng, bt, gold_pool, augment=True)
        E = forward_batch(model, dt, dm, ct, device)
        logits = -E
        for b in range(len(bt)):  # mask padded candidate slots
            logits[b, nc[b] :] = -1e9
        loss = F.cross_entropy(logits, torch.zeros(len(bt), dtype=torch.long, device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0:
            vacc, vrank = eval_val()
            log.append(
                {
                    "step": step,
                    "loss": round(float(loss.item()), 4),
                    "val_top1_acc": round(vacc, 4),
                    "val_gold_rank_mean": round(vrank, 3),
                }
            )
            print(
                f"[train] step {step}/{args.steps} loss={loss.item():.4f} "
                f"val_top1={vacc:.4f} val_rank={vrank:.2f}",
                flush=True,
            )
            if vacc > best_val:
                best_val, bad = vacc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"[train] early stop at {step} (no val gain for {bad} evals)", flush=True)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    corpus_hash = hashlib.sha256(",".join(sorted(tasks)).encode()).hexdigest()[:16]
    torch.save(
        {
            "state_dict": model.state_dict(),
            "best_val_top1": best_val,
            "log": log,
            "train_names": train_names,
            "val_names": val_names,
            "n_params": n_params,
            "corpus_hash": corpus_hash,
            "config": vars(args),
            "seed": SEED,
        },
        CKPT,
    )
    dur = time.time() - t0
    print(
        f"[train] DONE in {dur:.0f}s best_val_top1={best_val:.4f} params={n_params} -> {CKPT}",
        flush=True,
    )
    return {
        "train_duration_s": round(dur, 1),
        "best_val_top1_acc": round(best_val, 4),
        "n_params": n_params,
        "n_train_tasks": len(train_names),
        "n_val_tasks": len(val_names),
        "n_train_examples": len(train_ex),
        "corpus_hash": corpus_hash,
        "train_log": log,
    }


# ----------------------------------------------------------------------------- evaluation
def _auroc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _pass(tasks, key, ks=(1, 2)):
    hits = {k: 0 for k in ks}
    for t in tasks:
        rc = [c["correct"] for c in sorted(t["cands"], key=key)]
        for k in ks:
            hits[k] += int(any(rc[:k]))
    return {f"pass@{k}": round(hits[k] / len(tasks), 4) for k in ks}


def _fit_logreg(X, y, l2=1.0, iters=400, lr=0.2):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    npos = max(1, int(y.sum()))
    cw = np.where(y == 1, n / (2.0 * npos), n / (2.0 * max(1, n - npos)))
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(b + X @ w, -30, 30)))
        err = cw * (p - y)
        b -= lr * err.sum() / cw.sum()
        w -= lr * (X.T @ err / cw.sum() + l2 * w / n)
    return w, b


def _grouped_loto_union(tasks, feat_fn):
    """Grouped leave-one-TASK-NAME-out logistic ranker over per-candidate features (label: gold=0).
    Returns per-candidate OOF scores written into c['_u']."""
    groups = {}
    for ti, t in enumerate(tasks):
        groups.setdefault(t["task"], []).append(ti)
    for gname, tis in groups.items():
        tr = [c for tj, t in enumerate(tasks) if tj not in tis for c in t["cands"]]
        Xtr = np.stack([feat_fn(c) for c in tr])
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        ytr = np.array([0.0 if c["correct"] else 1.0 for c in tr])
        w, b = _fit_logreg((Xtr - mu) / sd, ytr)
        for ti in tis:
            for c in tasks[ti]["cands"]:
                x = (feat_fn(c) - mu) / sd
                c["_u"] = float(b + x @ w)


def evaluate(args, device, train_meta):
    t0 = time.time()
    ck = torch.load(CKPT, map_location="cpu")
    model = TransitionEBM().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    with gzip.open(POOL, "rt") as f:
        pool = json.load(f)
    entries = pool["entries"]
    # no-oracle structural audit (REAL checks — the first version of these was dead code
    # `assert X or True`, caught by the adversarial round; corrigendum_2026_06_09_stage2):
    # (a) entries/candidates carry ONLY the whitelisted keys; (b) no top-level gold field exists.
    # Demo pairs legitimately contain "output" grids (public context); the TEST gold grid must not
    # appear anywhere, which the whitelist enforces structurally (candidates have no gold/output key).
    _ENTRY_KEYS = {"task", "demos", "test_input", "candidates"}
    _CAND_KEYS = {"votes", "q_mean", "correct", "grid"}
    for e in entries:
        assert set(e.keys()) == _ENTRY_KEYS, f"unexpected entry keys: {set(e.keys()) - _ENTRY_KEYS}"
        for c in e["candidates"]:
            assert set(c.keys()) == _CAND_KEYS, (
                f"unexpected candidate keys: {set(c.keys()) - _CAND_KEYS}"
            )
    # score every candidate: E(cand | test_input, demos) — ONE forward per candidate grid, no pooling
    tasks = []
    with torch.no_grad():
        for e in entries:
            demos = e["demos"][:MAX_DEMOS]
            dt = torch.zeros(1, MAX_DEMOS, 2 * (N_COLORS + 1), MAX_HW, MAX_HW)
            dm = torch.zeros(1, MAX_DEMOS)
            for j, p in enumerate(demos):
                dt[0, j] = encode_pair(p["input"], p["output"])
                dm[0, j] = 1.0
            de = model.enc(dt[0, : len(demos)].to(device))
            rule = de.mean(0, keepdim=True)
            cands = []
            grids = [c["grid"] for c in e["candidates"]]
            Es = []
            for i0 in range(0, len(grids), 256):
                ct = torch.stack([encode_pair(e["test_input"], g) for g in grids[i0 : i0 + 256]])
                ce = model.enc(ct.to(device))
                Es.append(model.energy(rule.expand(len(ce), -1), ce).cpu().numpy())
            Es = np.concatenate(Es)
            tot_votes = sum(c["votes"] for c in e["candidates"])
            for c, E in zip(e["candidates"], Es):
                g = np.asarray(c["grid"])
                cands.append(
                    {
                        "votes": c["votes"],
                        "q_mean": c["q_mean"],
                        "correct": c["correct"],
                        "E": float(E),
                        "vote_share": c["votes"] / max(1, tot_votes),
                        "size": int(g.size),
                        "changed": float(
                            (clip_grid(c["grid"]).shape != clip_grid(e["test_input"]).shape)
                            or not np.array_equal(clip_grid(c["grid"]), clip_grid(e["test_input"]))
                        ),
                    }
                )
            tasks.append({"task": e["task"], "cands": cands})

    n = len(tasks)
    n_oracle = sum(1 for t in tasks if any(c["correct"] for c in t["cands"]))
    oracle2 = round(n_oracle / n, 4)

    # residual controls: global OLS + within-task vote-rank residual (the fairer corrigendum variant)
    allc = [c for t in tasks for c in t["cands"]]
    vs = np.array([c["votes"] for c in allc], float)
    es = np.array([c["E"] for c in allc], float)
    beta = np.cov(vs, es)[0, 1] / (vs.var() or 1.0)
    alpha = es.mean() - beta * vs.mean()
    for c in allc:
        c["_E_resid_global"] = c["E"] - (alpha + beta * c["votes"])
    for t in tasks:
        cs = t["cands"]
        vr = np.argsort(np.argsort([-c["votes"] for c in cs]))
        er = np.argsort(np.argsort([c["E"] for c in cs]))
        for c, a, b in zip(cs, er, vr):
            c["_E_rank_resid"] = float(a - b)

    # union baselines (grouped LOTO): no-latent control vs +E value-add
    _grouped_loto_union(
        tasks, lambda c: np.array([np.log1p(c["votes"]), c["vote_share"], c["q_mean"]])
    )
    for t in tasks:
        for c in t["cands"]:
            c["_union_noE"] = c["_u"]
    _grouped_loto_union(
        tasks, lambda c: np.array([np.log1p(c["votes"]), c["vote_share"], c["q_mean"], c["E"]])
    )
    for t in tasks:
        for c in t["cands"]:
            c["_union_withE"] = c["_u"]

    rankers = {
        "TRM_VOTE": lambda c: (-c["votes"],),
        "Q_MEAN": lambda c: (-c["q_mean"], -c["votes"]),
        "EBM_ENERGY": lambda c: (c["E"],),
        "EBM_ENERGY_vote_tiebreak": lambda c: (c["E"], -c["votes"]),
        "EBM_residual_over_vote_global": lambda c: (c["_E_resid_global"], -c["votes"]),
        "EBM_rank_residual_within_task": lambda c: (c["_E_rank_resid"], -c["votes"]),
        "HYBRID_vote_then_E": lambda c: (-c["votes"], c["E"]),
        "UNION_votes_qmean_voteshare": lambda c: (c["_union_noE"], -c["votes"]),
        "UNION_plus_E": lambda c: (c["_union_withE"], -c["votes"]),
    }
    res = {name: _pass(tasks, key) for name, key in rankers.items()}

    # within-task AUROC: macro + pair-weighted + hard-negatives (votes>=5) — per corrigendum
    def auroc_suite(score_fn):
        per, wsum, wtot = [], 0.0, 0.0
        hard = []
        for t in tasks:
            g = [score_fn(c) for c in t["cands"] if c["correct"]]
            ng = [score_fn(c) for c in t["cands"] if not c["correct"]]
            au = _auroc(g, ng)
            if au is not None:
                per.append(au)
                wsum += au * len(g) * len(ng)
                wtot += len(g) * len(ng)
            ngh = [score_fn(c) for c in t["cands"] if not c["correct"] and c["votes"] >= 5]
            auh = _auroc(g, ngh)
            if auh is not None:
                hard.append(auh)
        return (
            round(float(np.mean(per)), 4) if per else None,
            round(wsum / wtot, 4) if wtot else None,
            round(float(np.mean(hard)), 4) if hard else None,
        )

    e_macro, e_pairw, e_hard = auroc_suite(lambda c: -c["E"])
    v_macro, _, _ = auroc_suite(lambda c: c["votes"])

    # A1 shortcut diagnostics: what simple covariates does E read?
    def _sp(a, b):
        ar = np.argsort(np.argsort(a)).astype(float)
        br = np.argsort(np.argsort(b)).astype(float)
        ar -= ar.mean()
        br -= br.mean()
        d = np.linalg.norm(ar) * np.linalg.norm(br)
        return float(ar @ br / d) if d else 0.0

    within_sp = [
        _sp(np.array([c["E"] for c in t["cands"]]), np.array([-c["votes"] for c in t["cands"]]))
        for t in tasks
        if len(t["cands"]) > 2
    ]
    shortcuts = {
        "pearson_E_votes": round(float(np.corrcoef(es, vs)[0, 1]), 4),
        "pearson_E_logvotes": round(float(np.corrcoef(es, np.log1p(vs))[0, 1]), 4),
        "pearson_E_gridsize": round(
            float(np.corrcoef(es, np.array([c["size"] for c in allc], float))[0, 1]), 4
        ),
        "mean_within_task_spearman_E_vs_voterank": round(float(np.mean(within_sp)), 4),
    }

    # bootstrap CIs over tasks (deterministic LCG, B=1000): E vs vote; union+E vs union
    def _lcg(seed):
        x = seed
        while True:
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            yield x

    def _boot(kA, kB):
        gen, B, deltas = _lcg(SEED), 1000, []

        def p2(sample, key):
            return sum(
                int(any(c["correct"] for c in sorted(t["cands"], key=key)[:2])) for t in sample
            ) / len(sample)

        for _ in range(B):
            samp = [tasks[next(gen) % n] for _ in range(n)]
            deltas.append(p2(samp, kA) - p2(samp, kB))
        deltas.sort()
        return [round(deltas[25], 4), round(deltas[974], 4)]

    ci_E_vote = _boot(rankers["EBM_ENERGY"], rankers["TRM_VOTE"])
    ci_unionE_union = _boot(rankers["UNION_plus_E"], rankers["UNION_votes_qmean_voteshare"])

    vote2 = res["TRM_VOTE"]["pass@2"]
    e2 = res["EBM_ENERGY"]["pass@2"]
    headroom = (e2 - vote2) / max(1e-9, oracle2 - vote2)
    gates = {
        "selection_beats_vote": bool(e2 > vote2),
        "discrimination_macro_auroc_gt_0p70": bool((e_macro or 0) > 0.70),
        "discrimination_hard_neg_auroc": e_hard,
        "coverage_ge_0p80": True,
        "headroom_capture_ge_0p30": bool(headroom >= 0.30),
        "headroom_capture_fraction": round(headroom, 4),
        "union_value_add": bool(
            res["UNION_plus_E"]["pass@2"] > res["UNION_votes_qmean_voteshare"]["pass@2"]
        ),
    }
    resid_beats_vote = (
        res["EBM_residual_over_vote_global"]["pass@2"] > vote2
        or res["EBM_rank_residual_within_task"]["pass@2"] > vote2
    )
    generator_independent_real = bool(gates["selection_beats_vote"] and resid_beats_vote)

    verdict = (
        "complete: gap3_stage2_transition_ebm_"
        + (
            "BEATS_vote_generator_independent"
            if generator_independent_real
            else (
                "beats_vote_but_vote_confounded"
                if gates["selection_beats_vote"]
                else "does_not_beat_vote"
            )
        )
        + f"_n{n}_vote_{vote2}_ebm_{e2}_macroauroc_{e_macro}_hardauroc_{e_hard}"
        + f"_unionadd_{gates['union_value_add']}"
    )

    art = {
        "experiment": "arc3_gap3_stage2_transition_ebm",
        "title": "GAP-3 Stage 2: trained generator-independent ARC transition-EBM vs TRM frequency vote",
        "honest_verdict": verdict,
        "inference_substrate": "live_gpu_ebm_train_plus_offline_trm_candidate_rerank_no_oracle",
        "n_tasks": n,
        "n_oracle_hit": n_oracle,
        "oracle_pass2_ceiling": oracle2,
        "rankers": res,
        "auroc": {
            "ebm_macro": e_macro,
            "ebm_pair_weighted": e_pairw,
            "ebm_hard_negatives_votes_ge5": e_hard,
            "vote_macro_reference": v_macro,
        },
        "gates": gates,
        "generator_independent_signal_real": generator_independent_real,
        "bootstrap": {
            "ebm_vs_vote_pass2_ci95": ci_E_vote,
            "union_plus_E_vs_union_pass2_ci95": ci_unionE_union,
            "B": 1000,
        },
        "a1_shortcut_diagnostics": shortcuts,
        "a2_overfit_audit": {
            "train_val_top1_acc_vs_synthetic_negatives": train_meta.get("best_val_top1_acc"),
            "note": (
                "val measures gold-vs-SYNTHETIC-corruption top-1 on held-out TRAINING tasks; the "
                "eval pool is real TRM candidates on the ARC-1 EVAL split — fully disjoint tasks "
                "(asserted at train time). A large val-vs-eval gap = distribution shift between "
                "synthetic corruptions and real generator errors, not task leak."
            ),
        },
        "a3_no_oracle_audit": (
            "Energy computed from (demos, test_input, candidate_grid) only. The eval pool file does not "
            "contain gold output grids (exporter drops them); `correct` labels are used exclusively to "
            "score rankings post-hoc. Training corpus is arc-agi_training+concept; arc-agi_training2 was "
            "EXCLUDED after an overlap audit found 376/400 ARC-1 eval tasks inside it (29/30 pool tasks)."
        ),
        "training": train_meta,
        "model_specs": {
            "architecture": "PairEncoder CNN (4 conv blocks, GroupNorm, GELU) -> 256-d; rule = mean over "
            "demo-pair embeddings; energy head MLP([r,p,r*p,|r-p|])",
            "n_params": train_meta.get("n_params"),
            "negatives_per_positive": K_NEG,
            "negative_families": [
                "identity_copy",
                "demo_output_copy",
                "dihedral",
                "color_perm",
                "cell_noise",
                "row_col_resize",
                "shift",
                "other_task_output",
            ],
            "generator_provenance": "candidates from TRM arc_v1 step_518071 capped eval dump (eval only; "
            "no TRM data in training)",
        },
        "preconditions_checked": [
            {"resource": "cuda_available", "available": bool(torch.cuda.is_available())},
            {"resource": "eval_pool_export", "available": Path(POOL).exists()},
            {"resource": "kaggle_arc_corpus", "available": Path(KAGGLE).exists()},
        ],
        "random_seed": SEED,
        "reproducibility_checksum": train_meta.get("corpus_hash"),
        "honest_note": (
            "GAP-3 Stage 2. POSITIVE that survives the residual controls = a generator-independent "
            "content-aware energy reaches the present-but-mis-voted headroom -> register + re-confirm at "
            "400 scale (grouped folds). NEGATIVE = even a trained content energy cannot beat vote on this "
            "pool; the remaining moves are training on real generator errors (forfeits strict generator-"
            "independence) or richer context encoding. Reported as-is per FALSE_NEGATIVE_RISK; the "
            "positive control holds (oracle > vote). CAVEAT n=31; CI upper edge governs; re-confirm at "
            "400 before any irreversible claim."
        ),
        "duration_s": round(time.time() - t0 + train_meta.get("train_duration_s", 0.0), 1),
        "eval_duration_s": round(time.time() - t0, 1),
    }
    Path(ARTIFACT).write_text(json.dumps(art, indent=2, sort_keys=True) + "\n")
    print(f"-> {verdict}")
    for r in rankers:
        print(f"   {r:34s} pass@1={res[r]['pass@1']} pass@2={res[r]['pass@2']}")
    print(
        f"   oracle={oracle2} auroc: macro={e_macro} pairw={e_pairw} hard={e_hard} (vote ref {v_macro})"
    )
    print(f"   gates={gates}")
    print(f"   shortcuts={shortcuts}")
    print(f"   bootstrap E-vote CI95={ci_E_vote} | union+E - union CI95={ci_unionE_union}")
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
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device)
    train_meta = {}
    if args.mode in ("train", "all"):
        train_meta = train(args, device)
    if args.mode in ("eval", "all"):
        if not train_meta:
            ck = torch.load(CKPT, map_location="cpu")
            train_meta = {
                "best_val_top1_acc": ck.get("best_val_top1"),
                "n_params": ck.get("n_params"),
                "corpus_hash": ck.get("corpus_hash"),
                "train_duration_s": 0.0,
                "reused_checkpoint": True,
            }
        evaluate(args, device, train_meta)


if __name__ == "__main__":
    main()
