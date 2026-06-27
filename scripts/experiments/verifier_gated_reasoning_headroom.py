#!/usr/bin/env python3
"""HEADROOM check for VERIFIER-GATED REASONING (risk #2, before the full A/B/C build).

Question: does REASONING-ON beat REASONING-OFF on SimpleQA-Verified for the local model? The
arXiv:2603.09906 "Thinking to Recall" effect (reasoning unlocks parametric knowledge even on
single-hop QA) is the PREMISE of the whole verifier-gated-reasoning experiment: if reasoning gives
no recall gain for OUR model/corpus, there is nothing for a process verifier to protect (C approx B
by construction) and the full experiment is dead-on-arrival. This cheap paired check validates the
premise before the large build (phase-prototype + validation discipline).

Paired on identical items (same SimpleQA-Verified subset, GPU 1 Qwen3.5-9B):
- OFF: empty-think prefill, direct short answer (the de-risk's regime; ~9% acc there).
- ON : real <think> trace enabled, then a "Final answer:" line parsed out.
Decisive: acc(ON) - acc(OFF) with a paired-bootstrap CI95 EXCLUDING 0 -> headroom exists -> build
the full A/B/C experiment. CI95 straddling 0 -> no reasoning gain for this model/corpus -> the line
is moot here (report honestly; do NOT build).
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

# reuse the de-risk harness verbatim (same corpus loader, matcher, server client)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verifier_gated_reasoning_derisk import (  # noqa: E402
    SEED,
    _post,
    load_simpleqa,
)

REPO = Path(__file__).resolve().parents[2]
N_ITEMS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
N_BOOT = 2000


def answer_off(q: str) -> str:
    """Reasoning OFF: empty-think prefill, direct short answer."""
    prompt = (
        f"<|im_start|>user\nAnswer the question with ONLY the short factual answer, no explanation.\n"
        f"Question: {q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    d = _post({"prompt": prompt, "n_predict": 32, "temperature": 0.0, "cache_prompt": True})
    c = (d.get("content") or "").strip()
    return c.splitlines()[0].strip() if c else ""


def gen_facts(q: str) -> str:
    """Stage 1 of factual priming: elicit relevant intermediate facts (the paper's mechanism).
    A small model burns an unbounded <think> on hard QA, so cap it and use whatever was generated
    as the 'facts' (strip the <think> wrapper). These are the self-generated facts that may be
    hallucinated -- exactly what arm C's verifier will gate."""
    prompt = (
        f"<|im_start|>user\nList the key facts you can recall that are relevant to answering this "
        f"question. Be concise -- a few short factual statements, no final answer yet.\n"
        f"Question: {q}<|im_end|>\n<|im_start|>assistant\n"
    )
    d = _post(
        {"prompt": prompt, "n_predict": 256, "temperature": 0.0, "cache_prompt": True},
        timeout=120,
    )
    c = (d.get("content") or "")
    # strip a leading <think> wrapper; take the content up to </think> if it closed, else the cap
    if "<think>" in c:
        c = c.split("<think>", 1)[-1]
    c = c.split("</think>")[0]
    return c.strip()[:2000]


def answer_on(q: str) -> str:
    """Reasoning ON (factual priming, arm B): answer CONDITIONED on the self-generated facts."""
    facts = gen_facts(q)
    prompt = (
        f"<|im_start|>user\nRelevant facts:\n{facts}\n\nUsing these facts, answer the question with "
        f"ONLY the short factual answer, no explanation.\nQuestion: {q}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    d = _post({"prompt": prompt, "n_predict": 32, "temperature": 0.0, "cache_prompt": True})
    c = (d.get("content") or "").strip()
    return c.splitlines()[0].strip() if c else ""


def judge(q: str, gold: str, pred: str) -> int:
    """Local LLM-judge grading (SimpleQA-style), applied IDENTICALLY to both arms so grader bias is
    common-mode and cancels in the A-vs-B delta. Replaces crude substring matching, which had a
    length bias favoring the verbose priming arm (e.g. gold '6' substring-matches '1956')."""
    if not pred.strip():
        return 0
    prompt = (
        f"<|im_start|>user\nQuestion: {q}\nReference (correct) answer: {gold}\nProposed answer: {pred}\n"
        f"Is the proposed answer consistent with the reference answer (same fact; allow paraphrase, "
        f"formatting, and extra words)? Answer ONLY 'yes' or 'no'.<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    d = _post({"prompt": prompt, "n_predict": 1, "temperature": 0.0, "cache_prompt": True})
    c = (d.get("content") or "").strip().lower()
    return 1 if c.startswith("y") else 0


def boot_delta_ci95(off_ok, on_ok, n_boot: int, seed: int):
    """Paired bootstrap CI95 on acc(ON) - acc(OFF) over the same items."""
    rng = random.Random(seed)
    n = len(off_ok)
    idx = list(range(n))
    deltas = []
    for _ in range(n_boot):
        samp = [rng.choice(idx) for _ in idx]
        a_on = sum(on_ok[j] for j in samp) / n
        a_off = sum(off_ok[j] for j in samp) / n
        deltas.append(a_on - a_off)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    return round(lo, 4), round(hi, 4)


def main() -> int:
    started = time.time()
    items, schema = load_simpleqa(N_ITEMS, SEED)
    print(f"loaded {len(items)} simpleqa items (schema {schema})", flush=True)
    off_ok, on_ok = [], []
    for i, it in enumerate(items):
        try:
            a_off = answer_off(it["q"])
            a_on = answer_on(it["q"])
            off_ok.append(judge(it["q"], it["gold"], a_off))
            on_ok.append(judge(it["q"], it["gold"], a_on))
            if (i + 1) % 25 == 0:
                print(
                    f"  [{i+1}/{len(items)}] off={sum(off_ok)/len(off_ok):.3f} "
                    f"on={sum(on_ok)/len(on_ok):.3f}",
                    flush=True,
                )
        except Exception as exc:
            print(f"  [{i+1}] err {repr(exc)[:100]}", flush=True)
    n = len(off_ok)
    acc_off = sum(off_ok) / max(1, n)
    acc_on = sum(on_ok) / max(1, n)
    delta = acc_on - acc_off
    lo, hi = boot_delta_ci95(off_ok, on_ok, N_BOOT, SEED + 13)
    # also report the discordant pairs (McNemar-style): ON-right/OFF-wrong vs OFF-right/ON-wrong
    on_only = sum(1 for o, n_ in zip(off_ok, on_ok) if n_ and not o)
    off_only = sum(1 for o, n_ in zip(off_ok, on_ok) if o and not n_)
    headroom = bool(n >= 100 and lo > 0.0)
    if headroom:
        verdict = (f"complete: reasoning_headroom_confirmed_delta_{round(delta,4)}_ci95_{lo}_{hi}"
                   "_build_full_abc_experiment")
    elif lo <= 0.0 <= hi:
        verdict = (f"complete: no_reasoning_headroom_delta_{round(delta,4)}_ci95_{lo}_{hi}"
                   "_straddles_zero_line_moot_for_this_model_corpus")
    else:
        verdict = (f"complete: reasoning_hurts_delta_{round(delta,4)}_ci95_{lo}_{hi}")
    art = {
        "experiment": "verifier_gated_reasoning_headroom",
        "schema": "carnot.verifier_gated_reasoning_headroom.v1",
        "honest_verdict": verdict,
        "question": ("does reasoning-on beat reasoning-off (the arXiv:2603.09906 recall effect) for "
                     "the local model on SimpleQA-Verified? (risk #2: the headroom premise of "
                     "verifier-gated reasoning)"),
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "model": "Qwen3.5-9B-MTP (GPU 1, port 8921)",
        "corpus": "google/simpleqa-verified eval",
        "n_items": n,
        "acc_reasoning_off": round(acc_off, 4),
        "acc_reasoning_on": round(acc_on, 4),
        "delta_on_minus_off": round(delta, 4),
        "delta_ci95": [lo, hi],
        "discordant_on_right_off_wrong": on_only,
        "discordant_off_right_on_wrong": off_only,
        "n_bootstrap": N_BOOT,
        "reasoning_headroom_confirmed": headroom,
        "interpretation": (
            "headroom=True (CI95 of acc(ON)-acc(OFF) excludes 0) -> reasoning unlocks recall for this "
            "model/corpus; a process verifier protecting intermediate facts has something to improve "
            "-> build the full A/B/C experiment. headroom=False (CI95 straddles 0) -> no reasoning "
            "gain here; verifier-gating cannot raise final accuracy by construction -> the line is "
            "moot for this model/corpus (report; do not build)."
        ),
        "random_seed": SEED,
        "duration_s": round(time.time() - started, 2),
    }
    payload = dict(art)
    payload["reproducibility_checksum"] = ""
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    (REPO / "results" / "verifier_gated_reasoning_headroom.json").write_text(json.dumps(art, indent=2) + "\n")
    print("\n=== VERDICT:", verdict)
    print(f"acc OFF={acc_off:.3f}  acc ON={acc_on:.3f}  delta={delta:+.3f}  CI95=[{lo},{hi}]")
    print(f"discordant: ON-only-right={on_only}  OFF-only-right={off_only}")
    print("-> results/verifier_gated_reasoning_headroom.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
