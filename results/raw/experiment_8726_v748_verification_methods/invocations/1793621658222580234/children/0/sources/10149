"""POWERED driver for the on-policy verifier-as-self-improvement-reward test (#1).

Settles the underpowered result (process_weighted swung -0.06..+0.167 across runs because
generation was UNSEEDED and the eval set was 30 Q). This driver enforces the
statistical-power discipline:

  * SEEDED generation (torch.manual_seed per seed) -> reproducible teaching signal
  * >=100 eval questions (600-Q GSM8K pool, 25% eval = 150)
  * >=3 seeds -> report mean +/- std of each regime's delta-vs-base, not a single number

Gates (a regime "teaches" only if ALL hold):
  * gold-control: gold arm >= base each seed (positive control the harness works)
  * truncation: max eval truncation < 0.05 each seed (no silent corruption)
  * lift: mean(delta) - std(delta) > 0  (lower 1-sigma band clears zero across seeds)

Promotes/kills the `outcome_verifier_math_proc_plus_sc` registry candidate.

  .venv/bin/python scripts/experiments/process_reward_weighted_sft_onpolicy_powered.py \
      --corpus data/gsm8k_powered_600.jsonl --seeds 0,1,2 --K 6 --temp 0.4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_reward_weighted_sft_onpolicy_draft import run as run_one  # noqa: E402

OUT = REPO_ROOT / "results" / "process_reward_weighted_sft_onpolicy_powered.json"
REGIMES = ["process_weighted", "sc_weighted", "process_plus_sc", "gold", "unweighted"]


def _acc(x):
    return x["acc"] if isinstance(x, dict) else (x or 0.0)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def run(corpus_path, seeds, K, temp, smoke=False, write=True):
    started = time.time()
    per_seed = []
    for s in seeds:
        seed_out = REPO_ROOT / "results" / f"process_reward_weighted_sft_onpolicy_seed{s}.json"
        print(f"\n===== SEED {s} =====", flush=True)
        art = run_one(smoke=smoke, seed=s, K=K, temp=temp, corpus_path=corpus_path,
                      out_path=seed_out, write=True)
        per_seed.append(art)
        # incremental aggregate write so a mid-run crash still leaves a partial summary
        if write:
            _write_summary(per_seed, corpus_path, seeds, K, temp, started, partial=True)

    art = _write_summary(per_seed, corpus_path, seeds, K, temp, started, partial=False, write=write)
    return art


def _write_summary(per_seed, corpus_path, seeds, K, temp, started, *, partial, write=True):
    rows = [a for a in per_seed if isinstance(a, dict) and "accuracy_by_regime" in a]
    deltas = {r: [] for r in REGIMES}
    bases, gold_ok, trunc_ok, gen_trunc = [], [], [], []
    for a in rows:
        res = a["accuracy_by_regime"]
        base = _acc(res.get("base"))
        bases.append(base)
        for r in REGIMES:
            if res.get(r) is not None:
                deltas[r].append(_acc(res[r]) - base)
        gold_ok.append(bool(a.get("gold_control_ok")))
        trunc_ok.append(bool(a.get("truncation_ok")))
        if a.get("generation_meta"):
            gen_trunc.append(a["generation_meta"].get("gen_truncation_rate"))

    summary = {}
    for r in REGIMES:
        d = deltas[r]
        summary[r] = {"mean_delta": round(_mean(d), 4), "std_delta": round(_std(d), 4),
                      "n_seeds": len(d), "per_seed": [round(x, 4) for x in d],
                      "lower_1sigma": round(_mean(d) - _std(d), 4)}

    all_gold_ok = bool(rows) and all(gold_ok)
    all_trunc_ok = bool(rows) and all(trunc_ok)

    def _teaches(r):
        sm = summary[r]
        return bool(sm["n_seeds"] >= 2 and sm["lower_1sigma"] > 0 and all_gold_ok and all_trunc_ok)

    teaching = {r: _teaches(r) for r in ("process_weighted", "sc_weighted", "process_plus_sc")}
    any_teaches = any(teaching.values())
    status = ("TEACHES" if any_teaches else
              "HARNESS_BROKEN_gold_below_base" if not all_gold_ok else
              "TRUNCATION_INVALID" if not all_trunc_ok else "no_clear_lift")
    pps = summary["process_plus_sc"]
    verdict = (f"complete: powered_onpolicy_sft_{status}"
               f"_base{round(_mean(bases),3)}_procplussc_d{pps['mean_delta']}pm{pps['std_delta']}"
               f"_gold_d{summary['gold']['mean_delta']}_nseed{len(rows)}")

    art = {
        "experiment": "process_reward_weighted_sft_onpolicy_powered",
        "title": "powered_onpolicy_verifier_self_improvement_reward",
        "honest_verdict": verdict,
        "inference_substrate": "live_gpu_onpolicy_generation_plus_lora_sft_plus_verifier_ensemble",
        "partial": partial,
        "corpus_path": str(corpus_path), "seeds": list(seeds), "K": K, "temperature": temp,
        "n_seeds_completed": len(rows),
        "mean_base_accuracy": round(_mean(bases), 4),
        "delta_by_regime": summary,
        "teaching_by_regime": teaching,
        "gold_control_ok_all_seeds": all_gold_ok,
        "truncation_ok_all_seeds": all_trunc_ok,
        "gen_truncation_rate_by_seed": gen_trunc,
        "status": status,
        "duration_s": round(time.time() - started, 1),
        "interpretation": (
            "TEACHES iff a verifier-weighted regime's per-seed mean delta-vs-base clears zero at "
            "the lower 1-sigma band AND gold>=base each seed (control) AND trunc<5% each seed. "
            "process_plus_sc is the registry candidate; sc_weighted isolates the cheap "
            "self-consistency signal; if sc_weighted teaches but process_weighted doesn't, the "
            "lift is self-consistency (free) not the verifier (the honest moat question)."
        ),
        "principle_random_seed": "seeded generation makes the teaching signal reproducible per seed",
    }
    if write:
        OUT.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    return art


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/gsm8k_powered_600.jsonl")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip() != ""]
    corpus = str(REPO_ROOT / args.corpus) if not Path(args.corpus).is_absolute() else args.corpus
    art = run(corpus, seeds, args.K, args.temp, smoke=args.smoke)
    print(f"\n-> {art['honest_verdict']}")
    print(f"   teaching_by_regime: {art['teaching_by_regime']}")
    print(f"   delta_by_regime: { {k: v['mean_delta'] for k,v in art['delta_by_regime'].items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
