"""PHASE D (outer-loop bootstrap): TRAIN an oracle-distinct verifier on MuSR traces, test beats-SC.

The decisive moat question the conductor's D1/D2 could not run: does a TRAINED oracle-distinct verifier
beat genuine tuned self-consistency on MuSR, where the cheap PROMPTED proxy failed (0.515-0.535 vs SC
0.585) but headroom is present (exp5015 genuine_headroom_present=True)?

Design (leakage-safe, oracle-distinct, B2-moat-lint-compliant):
  - Input: results/musr_traces/q*.json (K reasoning traces/question + per-candidate correctness label).
  - Feature: mean-pooled all-MiniLM-L6-v2 embedding of (question + candidate reasoning). The verifier is a
    LEARNED scorer over reasoning text -- it is NOT the executable oracle / answer key (verifier_is_oracle
    =False). The `correct` label is used ONLY to train the head on TRAIN folds; the head NEVER sees the
    held-out question's gold (k-fold over QUESTIONS, so all candidates of a held-out q are out-of-fold).
  - Selector: per held-out question, pick the candidate with the highest predicted-correct probability;
    verifier_answer = that candidate's answer. (It always selects -> abstain_rate 0, not a hidden SC.)
  - Baselines on the SAME regenerated candidates: SC = K-way majority vote; report the matched SC accuracy
    AND cross-check vs the exp5015 genuine tuned-SC (0.585). oracle@K = is any candidate correct (headroom).
  - Stats: paired McNemar (verifier vs SC per question), bootstrap CI95 on the accuracy delta.
Gate: verifier beats SC with CI95 excluding 0 AND headroom_present -> moat REALIZED (oracle-distinct).
Else -> honest bounded null (the trained verifier also cannot capture MuSR headroom). retire_if_same_verdict.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRACES = REPO / "results" / "musr_traces"
RESULT = REPO / "results" / "experiment_phase_d_musr_trained_verifier.json"
GENUINE_SC = 0.585  # exp5015 corrected tuned-SC baseline (tuned_k=1, headroom_present)
N_FOLDS = 5
SEED = 20260630


def _embed(texts, device="cuda"):
    import torch
    from transformers import AutoModel, AutoTokenizer
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            h = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean pool
            emb = torch.nn.functional.normalize(emb, dim=1)
            out.append(emb.cpu().numpy())
    return np.vstack(out)


def _sc_answer(cands):
    votes = Counter(c["answer"] for c in cands if c.get("answer"))
    return votes.most_common(1)[0][0] if votes else None


def main() -> int:
    from sklearn.linear_model import LogisticRegression

    qs = sorted(TRACES.glob("q*.json"))
    data = [json.loads(p.read_text()) for p in qs]
    data = [d for d in data if d.get("candidates") and d.get("gold")]
    if len(data) < 50:
        print(f"INSUFFICIENT traces ({len(data)}); regen not complete -- abort, no fabrication")
        return 1

    # flatten candidates, remember question index per row
    rows, qidx, texts = [], [], []
    for di, d in enumerate(data):
        for c in d["candidates"]:
            rows.append(c)
            qidx.append(di)
            texts.append(f"{d['question']}\n\n{c.get('reasoning','')[:2000]}")
    X = _embed(texts)
    y = np.array([int(r.get("correct", 0)) for r in rows])
    qidx = np.array(qidx)

    # k-fold OVER QUESTIONS (all candidates of a held-out q are out-of-fold => no leakage)
    rng = random.Random(SEED)
    order = list(range(len(data)))
    rng.shuffle(order)
    folds = [set(order[f::N_FOLDS]) for f in range(N_FOLDS)]

    pred_prob = np.zeros(len(rows))
    for f in range(N_FOLDS):
        test_q = folds[f]
        tr = np.array([qidx[i] not in test_q for i in range(len(rows))])
        te = ~tr
        if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
            pred_prob[te] = 0.5
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        pred_prob[te] = clf.predict_proba(X[te])[:, 1]

    # per-question selection
    verifier_correct, sc_correct, oracle_correct = [], [], []
    for di, d in enumerate(data):
        idxs = [i for i in range(len(rows)) if qidx[i] == di]
        cands = [rows[i] for i in idxs]
        probs = [pred_prob[i] for i in idxs]
        sel = cands[int(np.argmax(probs))]
        verifier_correct.append(int(sel.get("answer") == d["gold"]))
        sc_ans = _sc_answer(cands)
        sc_correct.append(int(sc_ans == d["gold"]))
        oracle_correct.append(int(any(c.get("answer") == d["gold"] for c in cands)))

    n = len(data)
    v_acc = sum(verifier_correct) / n
    sc_acc = sum(sc_correct) / n
    oracle_acc = sum(oracle_correct) / n
    delta = v_acc - sc_acc

    # McNemar (verifier vs SC)
    b = sum(1 for i in range(n) if verifier_correct[i] and not sc_correct[i])  # v right, sc wrong
    c = sum(1 for i in range(n) if not verifier_correct[i] and sc_correct[i])  # v wrong, sc right
    from math import comb
    nn = b + c
    mcnemar_p = 1.0 if nn == 0 else min(1.0, 2 * sum(comb(nn, k) * 0.5 ** nn for k in range(min(b, c) + 1)))

    # bootstrap CI95 on delta (paired over questions)
    rng2 = random.Random(SEED + 1)
    deltas = []
    for _ in range(2000):
        idx = [rng2.randrange(n) for _ in range(n)]
        deltas.append(sum(verifier_correct[i] for i in idx) / n - sum(sc_correct[i] for i in idx) / n)
    deltas.sort()
    ci = [round(deltas[int(0.025 * len(deltas))], 4), round(deltas[int(0.975 * len(deltas))], 4)]

    headroom_present = bool(oracle_acc - sc_acc >= 0.10)
    beats_sc = bool(ci[0] > 0)
    moat_realized = bool(beats_sc and headroom_present)
    verdict = (f"complete_phase_d_trained_verifier_{'BEATS' if beats_sc else 'does_not_beat'}_sc_"
               f"v{v_acc:.3f}_sc{sc_acc:.3f}_delta{delta:+.3f}_ci{ci[0]}_{ci[1]}"
               f"_moat_{'realized' if moat_realized else 'bounded_null'}")

    artifact = {
        "experiment": "phase_d_musr_trained_verifier",
        "n_questions": n, "k_candidates_mean": round(len(rows) / n, 2),
        "trained_verifier_accuracy": round(v_acc, 4),
        "sc_accuracy_matched": round(sc_acc, 4), "genuine_tuned_sc_ref": GENUINE_SC,
        "oracle_at_k_accuracy": round(oracle_acc, 4),
        "delta_vs_sc": round(delta, 4), "delta_ci95": ci, "mcnemar_p": round(mcnemar_p, 4),
        "verifier_beats_sc": beats_sc, "headroom_present": headroom_present,
        "moat_realized": moat_realized,
        "verifier_is_oracle": False,
        "oracle_distinctness_note": "LEARNED LogisticRegression head over all-MiniLM embeddings of (question+"
            "reasoning); k-fold over QUESTIONS so the head never sees the held-out gold. NOT the answer key.",
        "abstain_rate": 0.0,
        "tuned_sc_baseline_source": "genuine K-way majority vote on the SAME regenerated candidates; cross-ref exp5015 0.585",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "solve_provenance": "development_proxy", "read_game_source": False, "random_seed": SEED,
        "retire_if_same_verdict": True,
        "honest_verdict": verdict,
        "methodology_note": "Traces regenerated by exp_phase_d_musr_trace_regen.py (Qwen3.5-9B-MTP on GPU-1). "
            "A trained verifier that BEATS SC with CI95 excl 0 on a headroom-present oracle-distinct domain "
            "= moat realized (unblocks DiffusionGemma); else an honest bounded null (the trained verifier, "
            "like the prompted proxy, cannot capture MuSR headroom).",
    }
    RESULT.write_text(json.dumps(artifact, indent=2))
    print(json.dumps({k: artifact[k] for k in ("trained_verifier_accuracy", "sc_accuracy_matched",
          "oracle_at_k_accuracy", "delta_vs_sc", "delta_ci95", "mcnemar_p", "verifier_beats_sc",
          "headroom_present", "moat_realized", "honest_verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
