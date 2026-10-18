"""GAP-4 k=3 CHAIN-ARMS agreement experiment — the pre-registered next codex purchase from the
precision-fixes adversarial round (results/arc3_gap4_precision_fixes_adversarial_verify.json).

PRE-REGISTERED PROTOCOL (fixed BEFORE any codex call; this docstring is the registration):

  TASKS (12, the clean chain-feasible set): the 13 unique ARC-2 probe tasks whose <=3-iteration
  refactor chain produced a demo-perfect program (12 from the probe run + 6ffbe589 whose demo-perfect
  program was recovered by the sandbox-blacklist regrade), MINUS aa4ec2a5 (flagged content-identical
  ARC-1-eval reuse; the other flagged reuse, 16b78196, is not chain-feasible and so absent anyway):
    13e47133, 2b83f449, 2d0172a1, 446ef5d2, 58490d8a, 58f5dbd5,
    6e453dd6, 6ffbe589, 7b80bb43, 9aaea919, b10624e5, d8e07eb2

  ARMS (k=3 per task):
    arm0 = the probe's CHAINED program — the best demo-perfect program re-extractable from the
           probe's archived transcripts under the CURRENT (word-boundary) sandbox; zero new cost.
           (d8e07eb2 carries a known iter0-transcript provenance ambiguity; its FINAL program is
           used, and the caveat is recorded in the artifact.)
    arm1, arm2 = two FRESH, INDEPENDENT <=3-iteration refactor chains (same induce_program protocol
           as the probe: failure feedback between iterations, early stop at demo-perfect), timeout
           600s per call, full transcripts archived. Up to 72 new codex calls total.

  GATE: per test input, >=2 demo-perfect arms whose executed outputs are hash-identical -> the
  agreed output is the gated prediction. Agreement is computed between program outputs ONLY.

  PRE-REGISTERED ACCEPTANCE (from the panel synthesis): the agreement gate beats the 0.52
  single-program baseline at alpha=0.05 only if >=5 agreement events occur on clean entries AND all
  are gold (0.52^5 = 0.038). Fewer events, or any non-gold agreement, is reported as-is — an honest
  underpowered/negative outcome. SECONDARY (the chain-vs-variance confound, Fisher p=0.092 in the
  singles round): the fresh chains' per-arm demo-perfect rate replicates the probe chain's ~0.52-0.57
  (chain mechanism real) or falls to the singles' ~0.20-0.26 (probe rate was run-variance).

  SCORING: gold (arc-agi_evaluation2_solutions.json) is used POST-HOC for scoring only; prompts
  contain demos + test input only (transcripts archived, auditable).

  ~/trm_venv/bin/python scripts/experiments/arc3_gap4_arc2_chain_ensemble.py --workers 4
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import sys
from carnot.paths import repo_root

# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
CARNOT = str(repo_root())
sys.path.insert(0, f"{CARNOT}/scripts/experiments")

from arc3_gap3_stage2_transition_ebm import SEED, ghash  # noqa: E402
from arc3_gap4_rule_exec_verifier import (  # noqa: E402
    _extract_code,
    demo_fit,
    induce_program,
    safe_transform_from_code,
)

ARC2_POOL = f"{CARNOT}/results/arc3_gap4_arc2_eval_pool.json.gz"
PROBE_TRANSCRIPTS = f"{CARNOT}/results/arc3_gap4_arc2_transcripts"
CHAIN_TRANSCRIPTS = f"{CARNOT}/results/arc3_gap4_arc2_chain_transcripts"
ARTIFACT = f"{CARNOT}/results/arc3_gap4_arc2_chain_ensemble.json"

CLEAN_TASKS = [
    "13e47133",
    "2b83f449",
    "2d0172a1",
    "446ef5d2",
    "58490d8a",
    "58f5dbd5",
    "6e453dd6",
    "6ffbe589",
    "7b80bb43",
    "9aaea919",
    "b10624e5",
    "d8e07eb2",
]
EXCLUDED_REUSE = ["aa4ec2a5", "16b78196"]


def probe_chain_program(task, demos):
    """arm0: best demo-perfect program among the probe's archived transcripts for this task, compiled
    under the CURRENT sandbox (recovers blacklist false-rejections, e.g. 6ffbe589)."""
    best = None
    for tf in sorted(glob.glob(f"{PROBE_TRANSCRIPTS}/{task}_iter*.txt")):
        raw = open(tf).read().split("===== RAW OUTPUT =====", 1)[-1]
        code = _extract_code(raw)
        fn = safe_transform_from_code(code) if code else None
        if fn and demo_fit(fn, demos) >= 1.0:
            best = code
            break  # earliest demo-perfect iteration = the chain's stopping point
    return best


def run(workers=4, iters=3, timeout=600, n_fresh=2):
    t0 = time.time()
    with gzip.open(ARC2_POOL, "rt") as f:
        entries = json.load(f)["entries"]
    by_task = {}
    for e in entries:
        by_task.setdefault(e["task"], []).append(e)
    for t in CLEAN_TASKS:
        assert t in by_task, f"pre-registered task {t} missing from pool"
        assert t not in EXCLUDED_REUSE
    Path(CHAIN_TRANSCRIPTS).mkdir(parents=True, exist_ok=True)

    def _arms_for(task):
        ents = by_task[task]
        demos = ents[0]["demos"]
        arms = [
            {
                "source": "probe_chain",
                "code": probe_chain_program(task, demos),
                "n_calls": 0,
                "codex_seconds": 0.0,
            }
        ]
        for j in range(n_fresh):
            tdir = f"{CHAIN_TRANSCRIPTS}/arm{j + 1}"
            Path(tdir).mkdir(parents=True, exist_ok=True)
            rec = induce_program(
                task,
                demos,
                ents[0]["test_input"],
                iters=iters,
                timeout=timeout,
                transcripts_dir=tdir,
            )
            arms.append(
                {
                    "source": f"fresh_chain{j + 1}",
                    "code": rec["code"],
                    "n_calls": rec["n_calls"],
                    "codex_seconds": rec["codex_seconds"],
                    "history": rec["history"],
                }
            )
        for a in arms:
            fn = safe_transform_from_code(a["code"]) if a["code"] else None
            a["demo_fit"] = round(demo_fit(fn, demos), 4) if fn else None
            a["demo_perfect"] = bool(fn and a["demo_fit"] >= 1.0)
            a["preds"] = (
                [fn(e["test_input"]) for e in ents]
                if (fn and a["demo_perfect"])
                else [None] * len(ents)
            )
        per_input = []
        for k in range(len(ents)):
            outs = [a["preds"][k] for a in arms if a["demo_perfect"] and a["preds"][k] is not None]
            agreed = None
            if len(outs) >= 2:
                hs = [ghash(o) for o in outs]
                for h in set(hs):
                    if hs.count(h) >= 2:
                        agreed = outs[hs.index(h)]
                        break
            per_input.append(
                {
                    "n_demo_perfect_arms": sum(1 for a in arms if a["demo_perfect"]),
                    "n_outputs": len(outs),
                    "agreement": agreed is not None,
                    "agreed_pred": agreed.tolist() if agreed is not None else None,
                }
            )
        return {
            "task": task,
            "arms": [{k2: v for k2, v in a.items() if k2 != "preds"} for a in arms],
            "per_input": per_input,
            "codex_seconds": round(sum(a["codex_seconds"] for a in arms), 1),
            "n_calls": sum(a["n_calls"] for a in arms),
        }

    print(
        f"[chain-ens] k=1+{n_fresh} chain arms on {len(CLEAN_TASKS)} clean tasks "
        f"(iters<={iters}, timeout={timeout}s, workers={workers})",
        flush=True,
    )
    recs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rec in ex.map(_arms_for, CLEAN_TASKS):
            recs.append(rec)
            npf = sum(1 for a in rec["arms"] if a["demo_perfect"])
            agr = sum(1 for pi in rec["per_input"] if pi["agreement"])
            print(
                f"  {rec['task']}: demo-perfect arms {npf}/3, agreement {agr}/"
                f"{len(rec['per_input'])} inputs ({rec['n_calls']} calls/"
                f"{rec['codex_seconds']}s)",
                flush=True,
            )

    # ---- post-hoc scoring (gold for SCORING only)
    sols = json.load(
        open("/home/ianblenke/trm_src/kaggle/combined/arc-agi_evaluation2_solutions.json")
    )
    ch = json.load(
        open("/home/ianblenke/trm_src/kaggle/combined/arc-agi_evaluation2_challenges.json")
    )

    def gold_for(task, test_input):
        for ti, pair in enumerate(ch[task]["test"]):
            if ghash(np.asarray(pair["input"])) == ghash(np.asarray(test_input)):
                return np.asarray(sols[task][ti])
        return None

    n_entries = n_agree = n_agree_gold = 0
    fresh_arm_perfect = fresh_arm_total = 0
    per_arm_gold = {"probe_chain": [0, 0], "fresh": [0, 0]}
    agreement_rows = []
    for rec in recs:
        ents = by_task[rec["task"]]
        for a in rec["arms"]:
            if a["source"] != "probe_chain":
                fresh_arm_total += 1
                fresh_arm_perfect += int(a["demo_perfect"])
            if a["demo_perfect"]:
                fn = safe_transform_from_code(a["code"])
                key = "probe_chain" if a["source"] == "probe_chain" else "fresh"
                for e in ents:
                    pred = fn(e["test_input"]) if fn else None
                    gold = gold_for(rec["task"], e["test_input"])
                    per_arm_gold[key][1] += 1
                    per_arm_gold[key][0] += int(
                        pred is not None and gold is not None and np.array_equal(pred, gold)
                    )
        for k, (pi, e) in enumerate(zip(rec["per_input"], ents)):
            n_entries += 1
            if pi["agreement"]:
                n_agree += 1
                gold = gold_for(rec["task"], e["test_input"])
                ok = gold is not None and np.array_equal(np.asarray(pi["agreed_pred"]), gold)
                n_agree_gold += int(ok)
                agreement_rows.append(
                    {"task": rec["task"], "input_idx": k, "agreed_is_gold": bool(ok)}
                )

    preregistered_pass = bool(n_agree >= 5 and n_agree_gold == n_agree)
    fresh_rate = round(fresh_arm_perfect / max(1, fresh_arm_total), 4)
    verdict = (
        "complete: gap4_chain_ensemble_"
        + (
            f"PREREG_PASS_{n_agree_gold}of{n_agree}_agreement_gold"
            if preregistered_pass
            else f"prereg_not_met_{n_agree_gold}of{n_agree}_agreement_gold"
        )
        + f"_freshchain_rate_{fresh_rate}_n{n_entries}_entries"
    )
    art = {
        "experiment": "arc3_gap4_arc2_chain_ensemble",
        "title": "GAP-4 k=3 chain-arms agreement gate (pre-registered) on clean ARC-2 tasks",
        "honest_verdict": verdict,
        "inference_substrate": "codex_program_induction_plus_offline_rerank_no_oracle",
        "preregistration": {
            "tasks": CLEAN_TASKS,
            "excluded_reuse_tasks": EXCLUDED_REUSE,
            "acceptance": ">=5 agreement events on clean entries AND all gold (0.52^5=0.038)",
            "secondary": "fresh-chain per-arm demo-perfect rate vs probe ~0.52-0.57 vs singles ~0.20-0.26",
            "registered_in": "this script's docstring, committed before the run",
        },
        "n_tasks": len(CLEAN_TASKS),
        "n_entries": n_entries,
        "agreement_entries": n_agree,
        "agreement_gold_entries": n_agree_gold,
        "precision_given_agreement": round(n_agree_gold / n_agree, 4) if n_agree else None,
        "preregistered_acceptance_met": preregistered_pass,
        "fresh_chain_arms_demo_perfect": f"{fresh_arm_perfect}/{fresh_arm_total}",
        "fresh_chain_per_arm_rate": fresh_rate,
        "per_arm_gold_given_perfect": {
            k: {"gold": v[0], "n": v[1], "rate": round(v[0] / v[1], 4) if v[1] else None}
            for k, v in per_arm_gold.items()
        },
        "agreement_rows": agreement_rows,
        "per_task": recs,
        "total_fresh_codex_calls": sum(r["n_calls"] for r in recs),
        "total_codex_seconds": round(sum(r["codex_seconds"] for r in recs), 1),
        "no_oracle_audit": (
            "Prompts = demos + test input only (fresh-chain transcripts archived in "
            "results/arc3_gap4_arc2_chain_transcripts/arm{1,2}/). Gold used post-hoc "
            "for scoring only; agreement is gold-free."
        ),
        "d8e07eb2_caveat": (
            "arm0 uses the task's final chained program; its iter0 transcript carries "
            "a known provenance ambiguity (recorded in the precision-fixes round)."
        ),
        "random_seed": SEED,
        "no_gpu_used": True,
        "duration_s": round(time.time() - t0, 1),
    }
    # clobber guard from birth: never overwrite an artifact that already has agreement data
    target = ARTIFACT
    if Path(ARTIFACT).exists():
        try:
            existing = json.loads(Path(ARTIFACT).read_text())
        except Exception:
            existing = {}
        if existing.get("agreement_entries") is not None and n_entries == 0:
            target = ARTIFACT.replace(".json", ".rerun.json")
    Path(target).write_text(json.dumps(art, indent=2, sort_keys=True) + "\n")
    print(f"\n-> {verdict}")
    print(
        f"   agreement {n_agree}/{n_entries} entries, gold {n_agree_gold}/{n_agree if n_agree else 1}; "
        f"prereg pass={preregistered_pass}"
    )
    print(f"   fresh-chain per-arm rate={fresh_rate} (probe chain ~0.52-0.57, singles ~0.20-0.26)")
    print(f"   per-arm gold|perfect: {art['per_arm_gold_given_perfect']}")
    print(f"   cost: {art['total_fresh_codex_calls']} calls / {art['total_codex_seconds']}s")
    return art


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    run(workers=a.workers, iters=a.iters, timeout=a.timeout)
