"""DRAFT (Phase 1): process-reward-weighted SFT -- does the verifier TEACH a generator?

The scale-up of the Sudoku #4 v4 result to reasoning, via the de-risked path
(dense per-step process-reward, outcome AUROC 0.73 -- NOT hard trace certification).
Off-policy first pass: reuse p01_gsm8k pre-generated samples (question + gold + 6
samples, ~49% correct) and fine-tune openbmb/MiniCPM5-1B (apache-2.0, Llama-arch,
1B, full-FT on a 3090) under three regimes, then compare held-out accuracy:

  base               -- no training (reference)
  process_weighted   -- SFT on (question->sample) WEIGHTED by the verifier's
                        process-reward (fraction-certified aggregate, NO gold)
  gold               -- SFT on gold-correct samples only (UPPER BOUND: uses labels)
  unweighted         -- SFT on all samples equally (control: no selection)

GATE: process_weighted > base (verifier signal teaches), and process_weighted
recovers a meaningful fraction of the gold lift. If process_weighted ~ unweighted,
the verifier adds nothing beyond more data; if ~base, it does not teach off-policy.

Off-policy = distilling p01-generator traces into MiniCPM (tests "can the verifier
SELECT good training data"); on-policy self-improvement (MiniCPM generates its own
traces) is the follow-up once this mechanism validates.

  # validate the harness (tiny) -- needs GPU, conductor paused:
  .venv/bin/python scripts/experiments/process_reward_weighted_sft_phase1_draft.py --smoke
  # full:
  .venv/bin/python scripts/experiments/process_reward_weighted_sft_phase1_draft.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GSM8K = REPO_ROOT / "data" / "p01_gsm8k_generations.jsonl"
OUT = REPO_ROOT / "results" / "process_reward_weighted_sft_phase1.json"
MODEL_ID = "openbmb/MiniCPM5-1B"
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _extract_answer(text: str) -> str | None:
    nums = _NUM.findall(str(text).replace(",", ""))
    return nums[-1] if nums else None


def _chunks(text: str) -> list[str]:
    body = _THINK.sub("", str(text)).strip()
    return [s.strip() for s in re.split(r"\n\s*\n", body)
            if len(s.strip()) >= 12 and re.search(r"[a-zA-Z0-9]", s)]


def _load_corpus(limit_q: int | None = None) -> list[dict]:
    rows = []
    with GSM8K.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            q = str(r.get("question") or "")
            gold = str(r.get("gold") or "").strip()
            samples = []
            for s in (r.get("samples") or []):
                txt = s if isinstance(s, str) else str(s.get("text") or "")
                ans = (s.get("answer") if isinstance(s, dict) else None) or _extract_answer(txt)
                if txt.strip():
                    samples.append({"text": txt, "gold_correct": int(str(ans).strip() == gold)})
            if q and gold and samples:
                rows.append({"question": q, "gold": gold, "samples": samples})
            if limit_q and len(rows) >= limit_q:
                break
    return rows


def _process_rewards(rows: list[dict]) -> None:
    """Attach a verifier process-reward (fraction-certified) to each sample in place."""

    from carnot.eval.verifier_error_independence_scissor_at_scale import (
        FoVerPanel, score_carnot_ensemble,
    )

    chunk_texts, owner = [], []  # owner = (row_idx, sample_idx)
    for ri, row in enumerate(rows):
        for si, s in enumerate(row["samples"]):
            for c in _chunks(s["text"]):
                chunk_texts.append(c)
                owner.append((ri, si))
    if not chunk_texts:
        return
    panel = FoVerPanel(
        rows=tuple({"idx": i} for i in range(len(chunk_texts))),
        labels=tuple(0 for _ in chunk_texts),
        texts=tuple(chunk_texts),
        panel_sha256=hashlib.sha256("".join(chunk_texts).encode("utf-8")).hexdigest(),
    )
    scoring = score_carnot_ensemble(panel, REPO_ROOT)
    pred_correct = [1 - int(p) for p in scoring.error_preds]
    agg: dict[tuple[int, int], list[int]] = {}
    for i, key in enumerate(owner):
        agg.setdefault(key, []).append(pred_correct[i])
    for ri, row in enumerate(rows):
        for si, s in enumerate(row["samples"]):
            preds = agg.get((ri, si), [])
            s["process_reward"] = (sum(preds) / len(preds)) if preds else 0.0


def _build_examples(rows: list[dict], regime: str) -> list[dict]:
    """(prompt, completion, weight) examples for an SFT regime."""

    out = []
    for row in rows:
        for s in row["samples"]:
            if regime == "gold" and not s["gold_correct"]:
                continue
            if regime == "process_weighted":
                w = float(s["process_reward"])
                if w <= 0.0:
                    continue
            else:
                w = 1.0
            out.append({"question": row["question"], "completion": s["text"], "weight": w})
    return out


# ----------------------------------------------------------------------------- GPU
def _run_training_and_eval(rows_tr, rows_ev, regimes, *, smoke: bool) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    max_len = 768 if not smoke else 384
    epochs = 1 if smoke else 2
    lr = 1e-5

    def _fmt(q: str) -> str:
        msgs = [{"role": "user", "content": f"Solve the problem. End with the final number.\n\n{q}"}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"Question: {q}\nSolution:"

    @torch.no_grad()
    def _eval(model) -> float:
        model.eval()
        hits = 0
        for row in rows_ev:
            ids = tok(_fmt(row["question"]), return_tensors="pt", truncation=True,
                      max_length=max_len).to(device)
            gen = model.generate(**ids, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            text = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            if _extract_answer(text) == row["gold"]:
                hits += 1
        return hits / len(rows_ev)

    def _train(examples) -> "object":
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)
        model.gradient_checkpointing_enable()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()
        steps = 0
        for _ in range(epochs):
            for ex in examples:
                prompt = _fmt(ex["question"])
                full = prompt + ex["completion"] + tok.eos_token
                enc = tok(full, return_tensors="pt", truncation=True, max_length=max_len).to(device)
                p_len = tok(prompt, return_tensors="pt", truncation=True,
                            max_length=max_len)["input_ids"].shape[1]
                labels = enc["input_ids"].clone()
                labels[0, :p_len] = -100  # train on the completion only
                out = model(**enc, labels=labels)
                loss = out.loss * float(ex["weight"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                steps += 1
        return model

    results = {}
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)
    results["base"] = _eval(base)
    del base
    torch.cuda.empty_cache()
    for regime in regimes:
        ex = _build_examples(rows_tr, regime)
        if not ex:
            results[regime] = None
            continue
        model = _train(ex)
        results[regime] = _eval(model)
        results[f"{regime}_n_examples"] = len(ex)
        del model
        torch.cuda.empty_cache()
    return results


def run(*, smoke: bool = False, seed: int = 0, write: bool = True) -> dict:
    import random
    started = time.time()
    rows = _load_corpus(limit_q=8 if smoke else None)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_ev = max(2, int(0.25 * len(rows)))
    rows_ev, rows_tr = rows[:n_ev], rows[n_ev:]

    # CPU: verifier process-rewards for the training samples.
    _process_rewards(rows_tr)

    try:
        import torch  # noqa: F401
        cuda = __import__("torch").cuda.is_available()
    except Exception:
        cuda = False
    if not cuda:
        art = {"experiment": "process_reward_weighted_sft_phase1_draft",
               "honest_verdict": "blocked_no_cuda",
               "inference_substrate": "none_blocked_preflight", "duration_s": time.time() - started}
        if write:
            OUT.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
        return art

    regimes = ["process_weighted", "gold", "unweighted"]
    res = _run_training_and_eval(rows_tr, rows_ev, regimes, smoke=smoke)

    base = res.get("base") or 0.0
    pw = res.get("process_weighted")
    gold = res.get("gold")
    d_pw = None if pw is None else round(pw - base, 4)
    d_gold = None if gold is None else round(gold - base, 4)
    frac = (None if (pw is None or gold is None or (gold - base) <= 1e-6)
            else round((pw - base) / (gold - base), 3))
    teaches = bool(pw is not None and pw - base >= 0.02)
    verdict = (f"complete: process_reward_sft_{'TEACHES' if teaches else 'no_clear_lift'}"
               f"_base{base:.3f}_pw{'na' if pw is None else round(pw,3)}"
               f"_gold{'na' if gold is None else round(gold,3)}_goldfrac{frac}")
    art = {
        "experiment": "process_reward_weighted_sft_phase1_draft",
        "title": "process_reward_weighted_sft",
        "honest_verdict": verdict,
        "inference_substrate": "live_gpu_full_finetune_minicpm5_1b_plus_verifier_ensemble",
        "model_id": MODEL_ID,
        "smoke": smoke,
        "n_train_questions": len(rows_tr),
        "n_eval_questions": len(rows_ev),
        "accuracy_by_regime": res,
        "delta_process_weighted": d_pw,
        "delta_gold_upper_bound": d_gold,
        "process_weighted_frac_of_gold_lift": frac,
        "gate": "process_weighted > base by >= 0.02 (verifier teaches off-policy)",
        "policy": "off_policy_reuses_p01_gsm8k_samples",
        "random_seed": seed,
        "duration_s": time.time() - started,
        "caveat": (
            "DRAFT, off-policy (distills p01-generator traces; tests verifier-as-training-data-"
            "selector). 1 seed, small GSM8K subset -- the ORDERING (pw vs base vs gold vs "
            "unweighted) is the result. On-policy self-improvement (MiniCPM generates its own "
            "traces) is the follow-up. Process-reward = fraction-certified aggregate (the 0.73-"
            "outcome-AUROC signal)."
        ),
    }
    if write:
        OUT.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    return art


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    art = run(smoke=args.smoke, seed=args.seed)
    print(f"-> {art['honest_verdict']}")
    print(f"   accuracy_by_regime: {art.get('accuracy_by_regime')}")
    return 0 if str(art["honest_verdict"]).startswith("complete:") else 1


if __name__ == "__main__":
    raise SystemExit(main())
