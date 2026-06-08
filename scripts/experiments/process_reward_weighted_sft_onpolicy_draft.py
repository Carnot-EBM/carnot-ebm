"""DRAFT (Phase 1 ON-POLICY): does the verifier TEACH a generator via self-improvement?

The clean version of the verifier-as-self-improvement-reward test. Off-policy was
structurally confounded (training on the p01 generator's CONCISE traces taught brevity,
not reasoning -> truncation asymmetry base 37% vs trained 3%). ON-POLICY removes that:
MiniCPM5-1B generates its OWN K traces per question (its own verbosity), the verifier
scores them, and we LoRA-SFT on its own process-reward-weighted traces -> base and trained
share style -> no truncation asymmetry -> the accuracy delta reflects REASONING.

  base               -- no training (reference)
  process_weighted   -- LoRA-SFT on MiniCPM's OWN traces WEIGHTED by verifier process-reward
  gold               -- LoRA-SFT on MiniCPM's OWN gold-correct traces only (UPPER BOUND)
  unweighted         -- LoRA-SFT on all MiniCPM's OWN traces equally (control)

All four disciplines ENFORCED (reusing the off-policy harness's training+instrumented
eval): headroom (GSM8K base ~0.49), gold-control gate (gold >= base), truncation guard
(<5%/arm, stop on <|im_end|>+</s>), multi-seed-ready.

  .venv/bin/python scripts/experiments/process_reward_weighted_sft_onpolicy_draft.py --smoke
  .venv/bin/python scripts/experiments/process_reward_weighted_sft_onpolicy_draft.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_reward_weighted_sft_phase1_draft import (  # noqa: E402
    MODEL_ID, _extract_answer, _load_corpus, _process_rewards, _run_training_and_eval,
)

OUT = REPO_ROOT / "results" / "process_reward_weighted_sft_onpolicy.json"


def _generate_on_policy(rows, *, K, temp, smoke, seed=0):
    """Replace each question's samples with K traces MiniCPM generates ITSELF."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # POWERED: seed generation so the teaching signal is reproducible per seed (the
    # statistical-power discipline: unseeded gen was why process_weighted swung
    # -0.06..+0.167 across runs). Seed varies the sampled traces between seeds.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    stop_ids = [tok.eos_token_id] + ([im_end] if isinstance(im_end, int) and im_end >= 0 else [])
    max_len = 384 if smoke else 768
    max_new = 512 if smoke else 1024  # MiniCPM is verbose; +repetition_penalty below

    def _fmt(q):
        msgs = [{"role": "user", "content": f"Solve the problem. End with the final number.\n\n{q}"}]
        try:
            # no-think: MiniCPM5 over-thinks GSM8K + rambles otherwise (100% trunc).
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except Exception:
            return f"Question: {q}\nSolution:"

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device)
    model.eval()
    t0 = time.time()
    n_trunc = n_gen = 0
    with torch.no_grad():
        for ri, row in enumerate(rows):
            ids = tok(_fmt(row["question"]), return_tensors="pt", truncation=True,
                      max_length=max_len).to(device)
            in_len = ids["input_ids"].shape[1]
            gen = model.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=temp,
                                 top_p=0.95, repetition_penalty=1.1, num_return_sequences=K,
                                 pad_token_id=tok.pad_token_id, eos_token_id=stop_ids)
            texts, answers = [], []
            for i in range(K):
                n_new = gen[i].shape[0] - in_len
                text = tok.decode(gen[i][in_len:], skip_special_tokens=True)
                texts.append(text)
                answers.append(str(_extract_answer(text)).strip())
                n_gen += 1
                if n_new >= max_new:
                    n_trunc += 1
            gold = str(row["gold"]).strip()
            samples = []
            for text, ans in zip(texts, answers):
                # SELF-CONSISTENCY: fraction of THIS question's K samples sharing this answer
                # (majority-agreement; the strong cheap outcome signal, free during RFT).
                sc = answers.count(ans) / len(answers) if answers else 0.0
                samples.append({"text": text, "gold_correct": int(ans == gold), "sc_agreement": sc})
            row["samples"] = samples
            if ri % 20 == 0:
                print(f"[gen] {ri}/{len(rows)} t={time.time()-t0:.0f}s "
                      f"gen_trunc={n_trunc/max(1,n_gen):.3f}", flush=True)
    del model
    torch.cuda.empty_cache()
    gen_trunc = n_trunc / max(1, n_gen)
    own_correct = sum(s["gold_correct"] for r in rows for s in r["samples"]) / max(1, n_gen)
    print(f"[gen] DONE {len(rows)} q x {K}; gen_truncation={gen_trunc:.3f} "
          f"own_sample_correct_rate={own_correct:.3f} t={time.time()-t0:.0f}s", flush=True)
    return {"gen_truncation_rate": round(gen_trunc, 4), "own_sample_correct_rate": round(own_correct, 4),
            "K": K, "temperature": temp}


def _load_questions(path):
    """Load question+gold rows directly (on-policy regenerates samples, so the
    placeholder sample in the file is only there to satisfy _load_corpus's contract)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            q, gold = str(r.get("question") or ""), str(r.get("gold") or "").strip()
            if q and gold:
                rows.append({"question": q, "gold": gold,
                             "samples": [{"text": f"The answer is {gold}", "answer": gold}]})
    return rows


def run(*, smoke=False, seed=0, K=6, temp=0.8, write=True, corpus_path=None, out_path=None):
    import torch
    started = time.time()
    rows = (_load_questions(corpus_path) if corpus_path
            else _load_corpus(limit_q=8 if smoke else None))
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_ev = max(2, int(0.25 * len(rows)))
    rows_ev, rows_tr = rows[:n_ev], rows[n_ev:]
    # eval rows only need question+gold; clear their p01 samples (unused on-policy).
    for r in rows_ev:
        r.pop("samples", None)

    if not torch.cuda.is_available():
        art = {"experiment": "process_reward_weighted_sft_onpolicy_draft",
               "honest_verdict": "blocked_no_cuda", "duration_s": time.time() - started}
        if write:
            (out_path or OUT).write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
        return art

    # ON-POLICY: MiniCPM generates its OWN training traces, then we score + train on them.
    gen_meta = _generate_on_policy(rows_tr, K=(2 if smoke else K), temp=temp, smoke=smoke, seed=seed)
    _process_rewards(rows_tr)  # verifier process-reward on MiniCPM's own traces

    regimes = ["process_weighted", "sc_weighted", "process_plus_sc", "gold", "unweighted"]
    res = _run_training_and_eval(rows_tr, rows_ev, regimes, smoke=smoke)

    def _acc(x):
        return x["acc"] if isinstance(x, dict) else (x or 0.0)
    base = _acc(res.get("base"))
    pw = None if res.get("process_weighted") is None else _acc(res["process_weighted"])
    gold = None if res.get("gold") is None else _acc(res["gold"])
    truncs = {k: v.get("truncation_rate") for k, v in res.items() if isinstance(v, dict)}
    max_trunc = max((t for t in truncs.values() if t is not None), default=0.0)
    trunc_ok = max_trunc <= 0.05
    gold_control_ok = bool(gold is not None and gold >= base - 0.02)
    d_pw = None if pw is None else round(pw - base, 4)
    d_gold = None if gold is None else round(gold - base, 4)
    frac = (None if (pw is None or gold is None or (gold - base) <= 1e-6)
            else round((pw - base) / (gold - base), 3))
    teaches = bool(pw is not None and pw - base >= 0.02 and gold_control_ok and trunc_ok)
    status = ("TEACHES" if teaches else "TRUNCATION_INVALID" if not trunc_ok else
              "HARNESS_BROKEN_gold_below_base" if not gold_control_ok else "no_clear_lift")
    verdict = (f"complete: onpolicy_process_reward_sft_{status}_base{base:.3f}"
               f"_pw{'na' if pw is None else round(pw,3)}_gold{'na' if gold is None else round(gold,3)}"
               f"_goldfrac{frac}_maxtrunc{round(max_trunc,3)}")
    art = {
        "experiment": "process_reward_weighted_sft_onpolicy_draft",
        "title": "onpolicy_process_reward_weighted_sft",
        "honest_verdict": verdict,
        "inference_substrate": "live_gpu_onpolicy_generation_plus_lora_sft_plus_verifier_ensemble",
        "model_id": MODEL_ID, "policy": "ON_POLICY_minicpm_generates_own_traces",
        "smoke": smoke, "n_train_questions": len(rows_tr), "n_eval_questions": len(rows_ev),
        "generation_meta": gen_meta,
        "accuracy_by_regime": res,
        "delta_process_weighted": d_pw, "delta_gold_upper_bound": d_gold,
        "process_weighted_frac_of_gold_lift": frac,
        "eval_truncation_rate_by_arm": truncs, "max_eval_truncation_rate": round(max_trunc, 4),
        "truncation_ok": trunc_ok, "gold_control_ok": gold_control_ok,
        "random_seed": seed, "duration_s": time.time() - started,
        "caveat": ("DRAFT, ON-POLICY (MiniCPM trains on its OWN verifier-scored traces -> no "
                   "off-policy style/truncation asymmetry). 1 seed, small corpus -> ORDERING is "
                   "the result. own_sample_correct_rate = the headroom the self-improvement "
                   "bootstraps from (must be >0 and <1). Gates: gold>=base, max_trunc<0.05."),
    }
    if write:
        (out_path or OUT).write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    return art


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.4)
    args = ap.parse_args()
    art = run(smoke=args.smoke, seed=args.seed, K=args.K, temp=args.temp)
    print(f"-> {art['honest_verdict']}")
    print(f"   generation: {art.get('generation_meta')}")
    print(f"   accuracy_by_regime: {art.get('accuracy_by_regime')}")
    return 0 if str(art["honest_verdict"]).startswith("complete:") else 1


if __name__ == "__main__":
    raise SystemExit(main())
