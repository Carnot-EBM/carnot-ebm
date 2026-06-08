"""DRAFT: MiniCPM5-1B BASE greedy accuracy per corpus -- the headroom precheck.

Phase 1's premise is that the corpus has HEADROOM (base model does NOT already solve
it). The GSM8K smoke showed base=1.0 (no headroom). hardmath's 19% "sample-correct-rate"
is the p01 GENERATOR's accuracy, NOT MiniCPM's -- so it does not establish MiniCPM's
headroom. This measures MiniCPM5-1B's OWN greedy accuracy per corpus before any training.
HEADROOM exists where base accuracy is materially below 1.0 (and above 0).
"""
from __future__ import annotations
import json, re, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPORA = {
    "hardmath": REPO_ROOT / "data" / "p01_hardmath_generations.jsonl",
    "gsm8k": REPO_ROOT / "data" / "p01_gsm8k_generations.jsonl",
}
MODEL_ID = "openbmb/MiniCPM5-1B"
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _ans(t: str):
    n = _NUM.findall(str(t).replace(",", ""))
    return n[-1] if n else None


def _load(p: Path):
    out = []
    with p.open() as f:
        for line in f:
            r = json.loads(line)
            q, g = str(r.get("question") or ""), str(r.get("gold") or "").strip()
            if q and g:
                out.append((q, g))
    return out


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
    model.eval()

    def fmt(q):
        msgs = [{"role": "user", "content": f"Solve the problem. End with the final number.\n\n{q}"}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"Question: {q}\nSolution:"

    results = {}
    t0 = time.time()
    for name, path in CORPORA.items():
        data = _load(path)
        hits = 0
        for q, g in data:
            ids = tok(fmt(q), return_tensors="pt", truncation=True, max_length=768).to("cuda")
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=512, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            txt = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            hits += (_ans(txt) == g)
        acc = hits / len(data)
        results[name] = {"n": len(data), "base_greedy_acc": round(acc, 4)}
        print(f"  {name:10s} n={len(data):4d}  base_greedy_acc={acc:.3f}  "
              f"headroom={'YES' if 0.05 < acc < 0.95 else ('NONE(too easy)' if acc>=0.95 else 'NONE(too hard)')}",
              flush=True)
    out = {"experiment": "minicpm_headroom_check", "model_id": MODEL_ID,
           "results": results, "duration_s": round(time.time() - t0, 1)}
    (REPO_ROOT / "results" / "minicpm_headroom_check.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("->", json.dumps(results))


if __name__ == "__main__":
    main()
