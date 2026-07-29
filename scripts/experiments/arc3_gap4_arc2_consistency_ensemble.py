"""GAP-4 precision fixes on the arc_v2/ARC-2 substrate — the two panel-ranked successors to the
exact-match gate, run as ONE experiment with a shared artifact.

PART A — GRADED MIN-HAMMING GATE (no codex; pure re-analysis of saved programs + pools).
The ARC-2 round found the one live rerank signal: a demo-perfect prediction 1/493 cells from gold
that the exact-match gate missed and the raw GRADED ranker caught. The successor design: promote the
argmin-hamming candidate iff demo_fit==1.0 AND min-hamming <= tau (else pure vote). The panel's
validation order is mandatory: confirm on the NON-degenerate ARC-1 venue that no vote-wins are lost
at pass@2 before crediting the ARC-2 recovery. Sweep tau in {0 (=exact gate), 0.005, 0.01, 0.02}.

PART B — k=3 INDEPENDENT-INDUCTION AGREEMENT GATE (codex; the precision fix).
ARC-2 demo-fit precision collapsed to ~0.47: half the demo-perfect programs are wrong on the test.
Micro-evidence from the probe's histories: independent demo-perfect pairs that AGREE on the test
output were gold 3/3. This part measures that properly: per ARC-2 task, k=3 INDEPENDENT single-shot
inductions — the probe's iter-0 program (re-extracted from its transcript: iter-0 is the only
chain-free call) + 2 FRESH single-shot codex calls (no failure feedback, timeout raised to 600s per
the panel: 300s killed all 3 iterations of an oracle-hit task). Gate: >=2 demo-perfect samples whose
test outputs AGREE. Measured: P(gold | agreement) vs the 0.47 single-program baseline, coverage cost,
and pass@3-style induction coverage (any-of-3 demo-perfect).

NO-ORACLE: prompts contain demos + test input only (transcripts archived, auditable); gold is used
post-hoc for SCORING only. Agreement is computed between PROGRAM OUTPUTS — never against gold.

  # Part A only (no codex):
  ~/trm_venv/bin/python scripts/experiments/arc3_gap4_arc2_consistency_ensemble.py --part a
  # full (Part A + Part B's ~46 codex calls):
  ~/trm_venv/bin/python scripts/experiments/arc3_gap4_arc2_consistency_ensemble.py --workers 4
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

from arc3_gap3_stage2_transition_ebm import SEED, _pass, ghash  # noqa: E402
from arc3_gap4_rule_exec_verifier import (  # noqa: E402
    _extract_code,
    ask_codex,
    demo_fit,
    induction_prompt,
    norm_hamming,
    safe_transform_from_code,
)

ARC1_POOL = f"{CARNOT}/results/arc3_gap3_stage2_eval_pool.json.gz"
ARC1_PROGRAMS = f"{CARNOT}/results/arc3_gap4_induced_programs.json"
ARC2_POOL = f"{CARNOT}/results/arc3_gap4_arc2_eval_pool.json.gz"
ARC2_PROGRAMS = f"{CARNOT}/results/arc3_gap4_arc2_induced_programs.json"
ARC2_TRANSCRIPTS = f"{CARNOT}/results/arc3_gap4_arc2_transcripts"
ENS_TRANSCRIPTS = f"{CARNOT}/results/arc3_gap4_arc2_ensemble_transcripts"
ARTIFACT = f"{CARNOT}/results/arc3_gap4_arc2_consistency_ensemble.json"
TAUS = (0.0, 0.005, 0.01, 0.02)


def load_pool(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)["entries"]


def attach_programs(entries, programs_path):
    """Re-execute each entry's SAVED program (None-safe) -> per-entry {demo_perfect, pred}."""
    progs = json.load(open(programs_path))["programs"]
    out = []
    for rec, e in zip(progs, entries):
        assert rec["task"] == e["task"]
        pred = None
        if rec["demo_perfect"] and rec["code"]:
            fn = safe_transform_from_code(rec["code"])
            pred = fn(e["test_input"]) if fn else None
        out.append({"demo_perfect": bool(rec["demo_perfect"]), "pred": pred})
    return out


# ------------------------------------------------------------------ Part A: graded threshold gate
def graded_gate_eval(entries, per_entry, taus=TAUS):
    """For each tau: promote argmin-hamming candidate iff demo-perfect AND min-hamming <= tau, else
    vote order. Returns per-tau pass@1/2 + win/loss accounting vs vote (pass@2)."""
    results = {}
    kv = lambda c: (-c["votes"],)  # noqa: E731
    tasks_v = [{"cands": e["candidates"]} for e in entries]
    vote = _pass(tasks_v, kv)
    vote_hits = {
        i
        for i, t in enumerate(tasks_v)
        if any(c["correct"] for c in sorted(t["cands"], key=kv)[:2])
    }
    for tau in taus:
        tasks = []
        n_fired = 0
        for e, pe in zip(entries, per_entry):
            cands = [dict(c) for c in e["candidates"]]
            promote_idx = None
            if pe["demo_perfect"] and pe["pred"] is not None:
                hams = [norm_hamming(c["grid"], pe["pred"]) for c in cands]
                mi = int(np.argmin(hams))
                if hams[mi] <= tau:
                    promote_idx = mi
                    n_fired += 1
            for j, c in enumerate(cands):
                c["_k"] = (0 if j == promote_idx else 1, -c["votes"])
            tasks.append({"cands": cands})
        key = lambda c: c["_k"]  # noqa: E731
        res = _pass(tasks, key)
        hits = {
            i
            for i, t in enumerate(tasks)
            if any(c["correct"] for c in sorted(t["cands"], key=key)[:2])
        }
        results[f"tau_{tau}"] = {
            "pass@1": res["pass@1"],
            "pass@2": res["pass@2"],
            "gate_fired": n_fired,
            "vote_wins_lost_p2": len(vote_hits - hits),
            "recovered_p2": len(hits - vote_hits),
        }
    results["vote_baseline"] = vote
    return results


# ------------------------------------------------- Part B: k=3 independent-induction agreement
def iter0_code(task):
    """The probe's chain-free first call, re-extracted from its archived transcript."""
    tf = f"{ARC2_TRANSCRIPTS}/{task}_iter0.txt"
    if not Path(tf).exists():
        return None
    raw = open(tf).read().split("===== RAW OUTPUT =====", 1)[-1]
    return _extract_code(raw)


def ensemble_for_task(task, demos, test_inputs, n_fresh=2, timeout=600):
    """k = 1 (probe iter-0) + n_fresh independent single-shot inductions. Per sample: demo_fit; if
    perfect, execute on each test input. Agreement = >=2 demo-perfect samples with identical output
    (per test input)."""
    samples = []
    c0 = iter0_code(task)
    samples.append({"source": "probe_iter0", "code": c0, "codex_s": 0.0})
    for j in range(n_fresh):
        tp = f"{ENS_TRANSCRIPTS}/{task}_fresh{j}.txt"
        raw, dt = ask_codex(induction_prompt(demos, test_inputs[0]), timeout, tp)
        samples.append({"source": f"fresh{j}", "code": _extract_code(raw), "codex_s": dt})
    for s in samples:
        fn = safe_transform_from_code(s["code"]) if s["code"] else None
        s["demo_fit"] = round(demo_fit(fn, demos), 4) if fn else None
        s["demo_perfect"] = bool(fn and s["demo_fit"] >= 1.0)
        s["preds"] = (
            [fn(ti) for ti in test_inputs]
            if (fn and s["demo_perfect"])
            else [None] * len(test_inputs)
        )
    per_input = []
    for k in range(len(test_inputs)):
        outs = [s["preds"][k] for s in samples if s["demo_perfect"] and s["preds"][k] is not None]
        agreed = None
        if len(outs) >= 2:
            hs = [ghash(o) for o in outs]
            for h in set(hs):
                if hs.count(h) >= 2:
                    agreed = outs[hs.index(h)]
                    break
        per_input.append(
            {
                "n_demo_perfect": sum(1 for s in samples if s["demo_perfect"]),
                "n_outputs": len(outs),
                "agreement": agreed is not None,
                "agreed_pred": agreed.tolist() if agreed is not None else None,
            }
        )
    return {
        "task": task,
        "samples": [
            {k: v for k, v in s.items() if k != "preds"}
            | {"code_len": len(s["code"]) if s["code"] else 0, "code": s["code"]}
            for s in samples
        ],
        "per_input": per_input,
        "codex_seconds": round(sum(s["codex_s"] for s in samples), 1),
        "n_fresh_calls": n_fresh,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=["a", "all"], default="all")
    ap.add_argument("--n_fresh", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0, help="cap unique tasks for a smoke")
    args = ap.parse_args()
    t0 = time.time()

    arc1_entries = load_pool(ARC1_POOL)
    arc2_entries = load_pool(ARC2_POOL)
    arc1_pe = attach_programs(arc1_entries, ARC1_PROGRAMS)
    arc2_pe = attach_programs(arc2_entries, ARC2_PROGRAMS)

    # ---- Part A
    part_a = {
        "arc1": graded_gate_eval(arc1_entries, arc1_pe),
        "arc2": graded_gate_eval(arc2_entries, arc2_pe),
    }
    arc1_safe_taus = [t for t in TAUS if part_a["arc1"][f"tau_{t}"]["vote_wins_lost_p2"] == 0]
    print(
        "[partA] ARC-1 graded-gate sweep (vote pass@2 "
        f"{part_a['arc1']['vote_baseline']['pass@2']}):"
    )
    for t in TAUS:
        r = part_a["arc1"][f"tau_{t}"]
        print(
            f"   tau={t}: pass@2={r['pass@2']} fired={r['gate_fired']} "
            f"lost={r['vote_wins_lost_p2']} recovered={r['recovered_p2']}"
        )
    print(
        "[partA] ARC-2 graded-gate sweep (vote pass@2 "
        f"{part_a['arc2']['vote_baseline']['pass@2']}):"
    )
    for t in TAUS:
        r = part_a["arc2"][f"tau_{t}"]
        print(
            f"   tau={t}: pass@2={r['pass@2']} fired={r['gate_fired']} "
            f"lost={r['vote_wins_lost_p2']} recovered={r['recovered_p2']}"
        )

    part_b = None
    if args.part == "all":
        Path(ENS_TRANSCRIPTS).mkdir(parents=True, exist_ok=True)
        by_task = {}
        for e in arc2_entries:
            by_task.setdefault(e["task"], []).append(e)
        tasks = sorted(by_task)
        if args.limit:
            tasks = tasks[: args.limit]
        print(
            f"[partB] k=1+{args.n_fresh} independent inductions on {len(tasks)} tasks "
            f"(timeout={args.timeout}s, workers={args.workers})",
            flush=True,
        )

        def _run(task):
            ents = by_task[task]
            return ensemble_for_task(
                task,
                ents[0]["demos"],
                [e["test_input"] for e in ents],
                n_fresh=args.n_fresh,
                timeout=args.timeout,
            )

        recs = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for rec in ex.map(_run, tasks):
                recs.append(rec)
                npf = sum(1 for s in rec["samples"] if s["demo_perfect"])
                agr = sum(1 for pi in rec["per_input"] if pi["agreement"])
                print(
                    f"  {rec['task']}: demo_perfect {npf}/3, agreement on "
                    f"{agr}/{len(rec['per_input'])} inputs ({rec['codex_seconds']}s)",
                    flush=True,
                )

        # ---- scoring (post-hoc, gold for SCORING only)
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

        n_entries = n_any_perfect = n_agree = n_agree_gold = 0
        n_single_perfect = n_single_gold = 0
        per_entry_rows = []
        for rec in recs:
            ents = by_task[rec["task"]]
            for k, (pi, e) in enumerate(zip(rec["per_input"], ents)):
                n_entries += 1
                gold = gold_for(rec["task"], e["test_input"])
                # single-program baseline: each demo-perfect sample counts once (per-sample precision)
                for s in rec["samples"]:
                    if s["demo_perfect"]:
                        fn = safe_transform_from_code(s["code"])
                        pred = fn(e["test_input"]) if fn else None
                        n_single_perfect += 1
                        n_single_gold += int(
                            pred is not None and gold is not None and np.array_equal(pred, gold)
                        )
                if pi["n_demo_perfect"] > 0:
                    n_any_perfect += 1
                if pi["agreement"]:
                    n_agree += 1
                    ok = gold is not None and np.array_equal(np.asarray(pi["agreed_pred"]), gold)
                    n_agree_gold += int(ok)
                    per_entry_rows.append(
                        {"task": rec["task"], "input_idx": k, "agreed_is_gold": bool(ok)}
                    )
        part_b = {
            "n_tasks": len(tasks),
            "n_entries": n_entries,
            "k_samples": 1 + args.n_fresh,
            "induction_any_of_k_entries": n_any_perfect,
            "agreement_entries": n_agree,
            "agreement_gold_entries": n_agree_gold,
            "precision_given_agreement": round(n_agree_gold / n_agree, 4) if n_agree else None,
            "per_sample_demo_perfect": n_single_perfect,
            "per_sample_gold": n_single_gold,
            "per_sample_precision": round(n_single_gold / n_single_perfect, 4)
            if n_single_perfect
            else None,
            "single_program_probe_baseline_precision": 0.47,
            "agreement_rows": per_entry_rows,
            "per_task": recs,
            "total_fresh_codex_calls": sum(r["n_fresh_calls"] for r in recs),
            "total_codex_seconds": round(sum(r["codex_seconds"] for r in recs), 1),
        }
        print(
            f"[partB] entries={n_entries} any-of-{1 + args.n_fresh}-perfect={n_any_perfect} "
            f"agreement={n_agree} agreement-gold={n_agree_gold} "
            f"P(gold|agree)={part_b['precision_given_agreement']} "
            f"per-sample precision={part_b['per_sample_precision']}",
            flush=True,
        )

    a1_best = max(arc1_safe_taus) if arc1_safe_taus else None
    verdict = (
        "complete: gap4_precision_fixes_"
        + (
            f"graded_gate_arc1_safe_up_to_tau_{a1_best}_"
            if a1_best is not None
            else "graded_gate_loses_arc1_votewins_"
        )
        + (
            f"agreement_precision_{part_b['precision_given_agreement']}"
            f"_vs_single_{part_b['per_sample_precision']}"
            if part_b
            else "partA_only"
        )
    )
    art = {
        "experiment": "arc3_gap4_arc2_consistency_ensemble",
        "title": "GAP-4 precision fixes: graded min-hamming gate (ARC-1-validated) + k=3 "
        "independent-induction agreement gate (ARC-2)",
        "honest_verdict": verdict,
        "inference_substrate": "codex_program_induction_plus_offline_rerank_no_oracle",
        "part_a_graded_gate": part_a,
        "part_a_note": (
            "Gate: promote argmin-hamming candidate iff demo_fit==1.0 AND min-hamming <= tau, else "
            "pure vote. Mandatory validation order per the ARC-2 panel: ARC-1 (non-degenerate venue) "
            "must lose ZERO vote-wins at pass@2 before the ARC-2 recovery is credited."
        ),
        "part_b_agreement": part_b,
        "part_b_note": (
            "k samples are INDEPENDENT: the probe's iter-0 call (chain-free, re-extracted from its "
            "archived transcript) + fresh single-shot calls with no failure feedback. Agreement is "
            "computed between program OUTPUTS only — never against gold; gold scores post-hoc."
        ),
        "no_oracle_audit": (
            "Prompts = demos + test input only (fresh-call transcripts archived in "
            "results/arc3_gap4_arc2_ensemble_transcripts/). Gold used exclusively for post-hoc "
            "scoring; the agreement gate is gold-free."
        ),
        "preconditions_checked": [
            {
                "resource": "arc1_pool+programs",
                "available": Path(ARC1_POOL).exists() and Path(ARC1_PROGRAMS).exists(),
            },
            {
                "resource": "arc2_pool+programs+transcripts",
                "available": Path(ARC2_POOL).exists()
                and Path(ARC2_PROGRAMS).exists()
                and Path(ARC2_TRANSCRIPTS).exists(),
            },
        ],
        "random_seed": SEED,
        "no_gpu_used": True,
        "duration_s": round(time.time() - t0, 1),
    }
    # CLOBBER GUARD (2026-06-10, post-adversarial-round): the original unconditional write let a
    # `--part a` re-run destroy a completed Part B record (7096 codex-seconds) — observed LIVE during
    # the review and restored from transcripts. A part-A-only result must never overwrite an artifact
    # that already carries Part B; it goes to a sidecar path instead.
    target = ARTIFACT
    if part_b is None and Path(ARTIFACT).exists():
        try:
            existing = json.loads(Path(ARTIFACT).read_text())
        except Exception:
            existing = {}
        if existing.get("part_b_agreement") is not None:
            target = ARTIFACT.replace(".json", ".partA_only.json")
            print(f"[guard] existing artifact has Part B — writing part-A-only result to {target}")
    Path(target).write_text(json.dumps(art, indent=2, sort_keys=True) + "\n")
    print(f"\n-> {verdict}")
    return art


if __name__ == "__main__":
    main()
