#!/usr/bin/env python3
"""De-risk for VERIFIER-GATED REASONING (the #1 risk, operator-directed 2026-06-27).

Question: can a CARNOT-SIDE, MODEL-NATIVE (no-retrieval) verifier signal discriminate the local
model's CORRECT vs HALLUCINATED factual answers? If yes, a process verifier can gate reasoning's
intermediate facts (the arXiv:2603.09906 "Thinking to Recall" fragility fix); if all model-native
signals are ~chance, the full experiment needs the retrieval-grounded variant instead.

Setup (mirrors the paper's self-generated-fact regime):
- Corpus: google/simpleqa-verified eval split (the paper's dataset), cached locally.
- Reasoner: local Qwen3.5-9B on GPU 1 (the outer-loop's dedicated 3090, port 8921) -- NO frontier API.
- For each question: get the model's greedy answer; label CORRECT/HALLUCINATED by normalized match to
  the gold answer. Then compute three model-native Carnot-side signals and measure AUROC of each at
  discriminating CORRECT from HALLUCINATED:
    (a) P(true)  -- self-eval "is this answer correct? yes/no", yes-token probability.
    (b) answer_logprob -- mean per-token logprob of the model's own answer (confidence).
    (c) self_consistency -- agreement rate of K sampled answers with the greedy answer.

Decisive read: if max AUROC >> 0.5 (and a permutation/label-shuffle control ~0.5), a model-native
Carnot verifier CAN discriminate factual hallucinations -> proceed to the full A/B/C verifier-gated
reasoning experiment. If all ~0.5 -> the #1 risk is real; the verifier must be retrieval-grounded.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PORT = 8921  # GPU-1 CUDA llama-server (Qwen3.5-9B)
SEED = 2603
# N from argv[1] (default 500). At ~8% accuracy, 500 items -> ~40 correct, clearing the project's
# N>=30 sample-size-rigor bar for the positive class (the n=120 prelim had only 10 correct).
N_ITEMS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
K_SC = 3  # self-consistency samples
N_BOOT = 2000  # paired-bootstrap resamples for the AUROC CI95
ARROW = (
    Path.home()
    / ".cache/huggingface/datasets/google___simpleqa-verified/simpleqa_verified/0.0.0"
    / "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc/simpleqa-verified-eval.arrow"
)


def load_simpleqa(n: int, seed: int):
    import pyarrow as pa

    tbl = None
    # HF arrow files are IPC STREAM format; try datasets.Dataset.from_file, then stream, then file.
    try:
        from datasets import Dataset

        tbl = Dataset.from_file(str(ARROW)).data.table
    except Exception:
        with pa.memory_map(str(ARROW), "r") as src:
            try:
                tbl = pa.ipc.open_stream(src).read_all()
            except Exception:
                src.seek(0)
                tbl = pa.ipc.open_file(src).read_all()
    cols = {c: tbl.column(c).to_pylist() for c in tbl.column_names}
    # find the question + answer columns (schema-robust)
    qkey = next((k for k in cols if k.lower() in ("problem", "question", "prompt")), None)
    akey = next((k for k in cols if k.lower() in ("answer", "gold_answer", "target", "solution")), None)
    if qkey is None or akey is None:
        raise RuntimeError(f"cannot find q/a columns in {tbl.column_names}")
    items = [{"q": str(q), "gold": str(a)} for q, a in zip(cols[qkey], cols[akey]) if q and a]
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n], tbl.column_names


def _post(payload: dict, timeout: int = 60) -> dict:
    payload.setdefault("stop", ["<|im_end|>"])
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _chat(user: str) -> str:
    """Qwen3 chat template + empty-think prefill (the reliable no-think trick on /completion)."""
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def model_answer(q: str):
    """Greedy answer + mean per-token logprob of the answer (confidence)."""
    prompt = _chat(
        "Answer the question with ONLY the short factual answer, no explanation.\n"
        f"Question: {q}"
    )
    d = _post({"prompt": prompt, "n_predict": 32, "temperature": 0.0, "n_probs": 1, "cache_prompt": True})
    ans = (d.get("content") or "").strip().splitlines()[0].strip() if d.get("content") else ""
    probs = d.get("completion_probabilities") or []
    lps = [p.get("logprob") for p in probs if isinstance(p.get("logprob"), (int, float))]
    mean_lp = sum(lps) / len(lps) if lps else -20.0
    return ans, mean_lp


def p_true(q: str, ans: str) -> float:
    """Self-eval: P(yes) that the proposed answer is correct."""
    prompt = _chat(
        "Question: " + q + "\nProposed answer: " + ans + "\n"
        "Is the proposed answer factually correct? Answer ONLY 'yes' or 'no'."
    )
    d = _post({"prompt": prompt, "n_predict": 1, "temperature": 0.0, "n_probs": 20})
    probs = (d.get("completion_probabilities") or [{}])[0].get("top_logprobs") or []
    pyes = pno = 0.0
    for t in probs:
        tok = (t.get("token") or "").strip().lower()
        p = math.exp(t.get("logprob", -50))
        if tok.startswith("yes"):
            pyes += p
        elif tok.startswith("no"):
            pno += p
    return pyes / (pyes + pno) if (pyes + pno) > 0 else 0.5


def self_consistency(q: str, greedy_ans: str, k: int) -> float:
    prompt = _chat(
        "Answer the question with ONLY the short factual answer, no explanation.\n"
        f"Question: {q}"
    )
    agree = 0
    for i in range(k):
        d = _post({"prompt": prompt, "n_predict": 32, "temperature": 0.7, "seed": SEED + i, "cache_prompt": True})
        a = (d.get("content") or "").strip().splitlines()[0].strip() if d.get("content") else ""
        if _norm(a) and (_norm(a) == _norm(greedy_ans) or _match(a, greedy_ans)):
            agree += 1
    return agree / k


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(the|a|an|of|in|on|at|is|was)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _match(a: str, gold: str) -> bool:
    na, ng = _norm(a), _norm(gold)
    if not na or not ng:
        return False
    return ng == na or ng in na or na in ng


def auroc(scores, labels) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def auroc_ci95(scores, labels, n_boot: int, seed: int):
    """Paired bootstrap CI95 on AUROC: resample (score,label) ROWS with replacement.
    A lower bound > 0.5 is the decisive 'discriminates above chance' test at this sample size."""
    rng = random.Random(seed)
    idx = list(range(len(labels)))
    boots = []
    for _ in range(n_boot):
        samp = [rng.choice(idx) for _ in idx]
        bs = [scores[j] for j in samp]
        bl = [labels[j] for j in samp]
        a = auroc(bs, bl)
        if a == a:  # not NaN (resample had both classes)
            boots.append(a)
    if not boots:
        return (float("nan"), float("nan"))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[min(len(boots) - 1, int(0.975 * len(boots)))]
    return (round(lo, 4), round(hi, 4))


def main() -> int:
    started = time.time()
    items, schema = load_simpleqa(N_ITEMS, SEED)
    print(f"loaded {len(items)} simpleqa items (schema {schema})", flush=True)
    rows = []
    for i, it in enumerate(items):
        try:
            ans, mean_lp = model_answer(it["q"])
            correct = 1 if _match(ans, it["gold"]) else 0
            pt = p_true(it["q"], ans)
            sc = self_consistency(it["q"], ans, K_SC)
            rows.append({"correct": correct, "p_true": pt, "answer_logprob": mean_lp, "self_consistency": sc})
            if (i + 1) % 20 == 0:
                acc = sum(r["correct"] for r in rows) / len(rows)
                print(f"  [{i+1}/{len(items)}] running acc={acc:.3f}", flush=True)
        except Exception as exc:
            print(f"  [{i+1}] err {repr(exc)[:100]}", flush=True)
    labels = [r["correct"] for r in rows]
    n_correct = sum(labels)
    n_wrong = len(labels) - n_correct
    signals = {
        "p_true": [r["p_true"] for r in rows],
        "answer_logprob": [r["answer_logprob"] for r in rows],
        "self_consistency": [r["self_consistency"] for r in rows],
    }
    aurocs = {k: round(auroc(v, labels), 4) for k, v in signals.items()}
    # paired-bootstrap CI95 per signal (decisive: lower bound > 0.5)
    aurocs_ci95 = {k: auroc_ci95(v, labels, N_BOOT, SEED + 11) for k, v in signals.items()}
    # label-shuffle control (must be ~0.5)
    rng = random.Random(SEED + 7)
    shuf = labels[:]
    rng.shuffle(shuf)
    aurocs_shuffled = {k: round(auroc(v, shuf), 4) for k, v in signals.items()}
    best = max((a for a in aurocs.values() if a == a), default=float("nan"))
    best_signal = max(aurocs, key=lambda k: aurocs[k] if aurocs[k] == aurocs[k] else -1)
    best_ci_lo = aurocs_ci95[best_signal][0]
    # DECISIVE gate (hardened): positive class >= 30 (sample-size rigor), best AUROC's CI95 lower
    # bound STRICTLY above chance, and clearly above the label-shuffle control.
    discriminates = bool(
        n_correct >= 30 and n_wrong >= 30
        and best_ci_lo == best_ci_lo and best_ci_lo > 0.5
        and best > (aurocs_shuffled[best_signal] + 0.1)
    )
    if n_correct < 30 or n_wrong < 30:
        verdict = (f"complete: derisk_underpowered_positive_class_correct{n_correct}_wrong{n_wrong}_need_30")
    elif discriminates:
        verdict = (f"complete: model_native_verifier_discriminates_hallucinations_{best_signal}_auroc_{best}"
                   f"_ci95lo_{best_ci_lo}_proceed_to_full_experiment")
    else:
        verdict = (f"complete: model_native_signals_at_chance_best_{best}_ci95lo_{best_ci_lo}"
                   "_retrieval_grounded_verifier_needed")
    art = {
        "experiment": "verifier_gated_reasoning_derisk",
        "schema": "carnot.verifier_gated_reasoning_derisk.v1",
        "honest_verdict": verdict,
        "question": ("can a Carnot-side model-native (no-retrieval) verifier discriminate the local "
                     "model's correct vs hallucinated factual answers? (the #1 risk of verifier-gated reasoning)"),
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "model": "Qwen3.5-9B-MTP (GPU 1, port 8921)",
        "corpus": "google/simpleqa-verified eval",
        "n_items": len(rows),
        "model_accuracy": round(n_correct / max(1, len(rows)), 4),
        "n_correct": n_correct,
        "n_hallucinated": n_wrong,
        "auroc_per_signal": aurocs,
        "auroc_ci95_per_signal": {k: list(v) for k, v in aurocs_ci95.items()},
        "auroc_shuffled_control": aurocs_shuffled,
        "best_signal": best_signal,
        "best_auroc": best,
        "best_auroc_ci95": list(aurocs_ci95[best_signal]),
        "n_bootstrap": N_BOOT,
        "model_native_verifier_discriminates": discriminates,
        "interpretation": (
            "discriminates=True -> a Carnot-side model-native verifier can gate hallucinated intermediate "
            "facts; proceed to the full A/B/C verifier-gated-reasoning experiment. discriminates=False -> "
            "model-native signals are ~chance; the verifier must be retrieval-grounded (the #1 risk)."
        ),
        "random_seed": SEED,
        "duration_s": round(time.time() - started, 2),
    }
    payload = dict(art)
    payload["reproducibility_checksum"] = ""
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    # n=120 was the prelim (preserved); the hardened run writes a distinct file.
    fname = "verifier_gated_reasoning_derisk.json" if len(rows) == 120 else "verifier_gated_reasoning_derisk_hardened.json"
    (REPO / "results" / fname).write_text(json.dumps(art, indent=2) + "\n")
    print("\n=== VERDICT:", verdict)
    print("model_accuracy:", art["model_accuracy"], "| n_correct:", n_correct, "n_wrong:", n_wrong)
    print("AUROC per signal:", aurocs)
    print("AUROC CI95 per signal:", {k: list(v) for k, v in aurocs_ci95.items()})
    print("shuffled control:", aurocs_shuffled)
    print(f"-> results/{fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
