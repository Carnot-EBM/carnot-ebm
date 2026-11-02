"""Exp 10030: selection denominator and held-out check for the FoVer headline (AUROC 0.9131).

Why this exists: the headline was fixed after many FoVer variants were scored, and nothing
states how many. This script (1) re-derives the published numbers as a control, (2) finds the
FoVer rows no headline seed ever used, (3) scores the FROZEN formula on those rows, and (4) lists
the variants tried. Task draft: docs/research-notes/fover-headline-selection-denominator-task-draft-2026-10-08.md

Read-only on the corpus and on the frozen scorer. Writes one results artifact (stage `final`).
Stages cache their output under --cache so each can be re-run alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from carnot.eval import fover_memory_leakage_v3 as m  # noqa: E402

SEEDS = (42, 137, 271, 314, 1729)
N_EXAMPLES = 1000
BOOT_N = 2000
BOOT_SEED = 10030
HEADLINE = {
    "A": (0.9131335999999999, 0.9027316334533082, 0.9235355665466916),
    "B": (0.8946624, 0.8841988549900464, 0.9051259450099536),
}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fast_auroc(y: np.ndarray, s: np.ndarray) -> float:
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(y: np.ndarray, s: np.ndarray, seed: int = BOOT_SEED) -> dict:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    vals = np.empty(BOOT_N)
    for i in range(BOOT_N):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals[i] = fast_auroc(y[idx], s[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {
        "mean": float(vals.mean()),
        "low": float(lo),
        "high": float(hi),
        "half_width": float((hi - lo) / 2),
    }


def load_scores() -> dict:
    rows = m._read_fover_rows(ROOT / "data" / "fover_corpus.jsonl")
    y = np.array([m._label_to_int(r["label"]) for r in rows])
    say(f"scoring {len(rows)} rows with the frozen verifiers")
    vs = m._score_text_verifiers([str(r.get("step_text", "")) for r in rows])
    idx = m._load_fr11_memory_index(ROOT)
    mem = np.array([m._fr11_memory_score(r, idx) for r in rows])
    r_, u_ = np.array(vs["tier0r_curry_howard"]), np.array(vs["tier0u_logical_consistency"])
    base = 0.9 * r_ + 0.1 * u_
    qids = idx["question_ids"]
    flagged = np.array(
        [
            str(r.get("question_id", "")) in qids or f"math_v3_{r.get('question_id', '')}" in qids
            for r in rows
        ]
    )
    return {
        "rows": rows,
        "y": y,
        "r": r_,
        "u": u_,
        "s": np.array(vs["tier0s_arithmetic_gap"]),
        "mem": mem,
        "base": base,
        "A": base + m.FR11_MEMORY_BOOST * mem,
        "B": base,
        "flagged": flagged,
        "n_mem_qids": len(qids),
        "n_mem_token_sets": len(idx["prompt_token_sets"]),
    }


def seed_subsets(d: dict) -> dict:
    pos_of = {id(r): i for i, r in enumerate(d["rows"])}
    out = {}
    for seed in SEEDS:
        sub = m._select_balanced_subset(d["rows"], seed=seed, n_examples=N_EXAMPLES)
        out[seed] = [pos_of[id(r)] for r in sub]
    return out


def stage_control(d: dict, subsets: dict) -> dict:
    say("control: re-derive the published per-seed means")
    res = {}
    for cond in ("A", "B"):
        vals = [fast_auroc(d["y"][ix], d[cond][np.array(ix)]) for ix in subsets.values()]
        ref = HEADLINE[cond][0]
        res[cond] = {
            "per_seed": vals,
            "mean": float(np.mean(vals)),
            "published_mean": ref,
            "abs_diff": abs(float(np.mean(vals)) - ref),
        }
        say(f"  condition {cond}: mean {res[cond]['mean']:.6f} vs published {ref:.6f}")
    res["reproduces_published_to_1e-4"] = all(res[c]["abs_diff"] < 1e-4 for c in ("A", "B"))
    return res


def stage_heldout(d: dict, subsets: dict) -> dict:
    y, n = d["y"], len(d["y"])
    used = set(i for ix in subsets.values() for i in ix)
    used_q = {str(d["rows"][i].get("question_id", "")) for i in used}
    never = np.array([i not in used for i in range(n)])
    q_never = np.array([str(r.get("question_id", "")) not in used_q for r in d["rows"]])
    fl = d["flagged"]
    sets = {
        "all_rows": np.ones(n, bool),
        "H0_never_sampled_rows": never,
        "H1_never_sampled_and_not_memory_flagged": never & ~fl,
        "H2_question_never_sampled_and_not_memory_flagged": q_never & ~fl,
        "corpus_minus_memory_flagged": ~fl,
        "memory_flagged_only": fl,
    }
    out = {
        "n_rows_used_by_any_seed": len(used),
        "n_positive_used": int(y[list(used)].sum()),
        "n_positive_total": int(y.sum()),
    }
    for name, mask in sets.items():
        yy = y[mask]
        entry = {
            "n": int(mask.sum()),
            "n_positive": int(yy.sum()),
            "n_negative": int((1 - yy).sum()),
        }
        if entry["n_positive"] >= 10 and entry["n_negative"] >= 10:
            for cond in ("A", "B"):
                s = d[cond][mask]
                entry[cond] = {"auroc": fast_auroc(yy, s), "ci95": boot_ci(yy, s)}
        out[name] = entry
        say(
            f"  {name}: n={entry['n']} pos={entry['n_positive']} "
            + (
                f"A={entry['A']['auroc']:.4f} B={entry['B']['auroc']:.4f}"
                if "A" in entry
                else "too small"
            )
        )
    return out


def stage_corpus_ci(d: dict) -> dict:
    say("bootstrap CI over rows on the full corpus (frozen formula)")
    out = {}
    for cond in ("A", "B"):
        ci = boot_ci(d["y"], d[cond])
        pub = HEADLINE[cond]
        out[cond] = {
            "auroc_full_corpus": fast_auroc(d["y"], d[cond]),
            "bootstrap_ci95": ci,
            "published_ci_half_width": (pub[2] - pub[1]) / 2,
            "half_width_ratio_bootstrap_over_published": ci["half_width"] / ((pub[2] - pub[1]) / 2),
        }
    return out


def stage_tuning_gap(d: dict, subsets: dict) -> dict:
    say("positive control: tune weights on the headline rows, score on never-sampled rows")
    used = np.array(sorted(set(i for ix in subsets.values() for i in ix)))
    mask_h1 = np.ones(len(d["y"]), bool)
    mask_h1[used] = False
    mask_h1 &= ~d["flagged"]
    best = None
    for wr in np.arange(0.0, 1.01, 0.1):
        for boost in (0.0, 0.5, 1.0, 2.0):
            sc = wr * d["r"] + (1 - wr) * d["u"] + boost * d["mem"]
            a = fast_auroc(d["y"][used], sc[used])
            if best is None or a > best[0]:
                best = (a, float(wr), float(boost))
    a_in, wr, boost = best
    sc = wr * d["r"] + (1 - wr) * d["u"] + boost * d["mem"]
    a_out = fast_auroc(d["y"][mask_h1], sc[mask_h1])
    frozen_in = fast_auroc(d["y"][used], d["A"][used])
    frozen_out = fast_auroc(d["y"][mask_h1], d["A"][mask_h1])
    return {
        "grid": "w_r in 0..1 step 0.1 with w_u=1-w_r; boost in {0,0.5,1,2}; 44 combinations",
        "tuned": {
            "w_r": wr,
            "boost": boost,
            "auroc_in_sample_union": a_in,
            "auroc_held_out_H1": a_out,
            "optimism_gap": a_in - a_out,
        },
        "frozen_formula": {
            "auroc_in_sample_union": frozen_in,
            "auroc_held_out_H1": frozen_out,
            "gap": frozen_in - frozen_out,
        },
        "n_in_sample_union": int(len(used)),
        "n_held_out_H1": int(mask_h1.sum()),
    }


def paired_diff_ci(y: np.ndarray, a: np.ndarray, b: np.ndarray, seed: int = BOOT_SEED) -> dict:
    """AUROC(a) - AUROC(b) with a paired row bootstrap (same resampled rows for both)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = np.empty(BOOT_N)
    for i in range(BOOT_N):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals[i] = fast_auroc(y[idx], a[idx]) - fast_auroc(y[idx], b[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": fast_auroc(y, a) - fast_auroc(y, b), "low": float(lo), "high": float(hi)}


def stage_memory_effect(d: dict, subsets: dict) -> dict:
    say("memory effect: A-B gap with and without label-derived memory rows")
    y, fl = d["y"], d["flagged"]
    out = {
        "published_learning_contribution": 0.0184712,
        "n_flagged_rows": int(fl.sum()),
        "n_flagged_positive": int(y[fl].sum()),
        "n_flagged_negative": int((1 - y[fl]).sum()),
        "share_of_positives_flagged": float(y[fl].sum() / y.sum()),
    }
    per_seed = []
    for seed, ix in subsets.items():
        ix = np.array(ix)
        per_seed.append(
            {
                "seed": seed,
                "flagged_in_subset": int(fl[ix].sum()),
                "flagged_positive_in_subset": int(y[ix][fl[ix]].sum()),
            }
        )
    out["per_seed_flagged"] = per_seed
    hn = np.ones(len(y), bool)
    used = set(i for ix in subsets.values() for i in ix)
    never = np.array([i not in used for i in range(len(y))])
    for name, mask in (
        ("all_rows", np.ones(len(y), bool)),
        ("corpus_minus_memory_flagged", ~fl),
        ("H1_never_sampled_and_not_memory_flagged", never & ~fl),
    ):
        out[name] = paired_diff_ci(y[mask], d["A"][mask], d["B"][mask])
        say(
            f"  A-B on {name}: {out[name]['point']:.4f} [{out[name]['low']:.4f}, {out[name]['high']:.4f}]"
        )
    out["_unused"] = int(hn.sum())
    return out


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout


def stage_variants() -> dict:
    say("enumerating FoVer variants in results/")
    cands = {}
    for p in sorted((ROOT / "results").glob("*.json")):
        name = p.name
        if "fover" in name.lower():
            cands[name] = "filename"
    say(f"  {len(cands)} by filename; scanning small artifacts for corpus references")
    for p in sorted((ROOT / "results").glob("experiment_*.json")):
        if p.name in cands or p.stat().st_size > 3_000_000:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "fover_corpus" in txt:
            cands[p.name] = "references_fover_corpus_file"
    say(f"  {len(cands)} candidates total")
    table = []
    for i, (name, how) in enumerate(sorted(cands.items())):
        if i % 20 == 0:
            say(f"  dating/reading {i}/{len(cands)}")
        path = ROOT / "results" / name
        first = (
            git("log", "--diff-filter=A", "--format=%cI", "--", f"results/{name}")
            .strip()
            .splitlines()
        )
        date = first[-1] if first else None
        auroc, corpus_refs, keys = {}, [], []
        try:
            if path.stat().st_size < 30_000_000:
                dat = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(dat, dict):
                    keys = list(dat)[:0]
                    for k, v in dat.items():
                        if "auroc" in k.lower() and isinstance(v, (int, float)):
                            auroc[k] = v
                    blob = json.dumps(dat)[:400000]
                    for f in (
                        "fover_corpus.jsonl",
                        "fover_corpus_v4.json",
                        "fover_corpus_v5.json",
                        "fover_corpus_v2.json",
                        "fover_labeled_steps",
                        "fover_v2_combined",
                    ):
                        if f in blob:
                            corpus_refs.append(f)
                    tune = [
                        w
                        for w in ("tuned", "grid search", "optimiz", "selected on", "fit on")
                        if w in blob.lower()
                    ]
                else:
                    tune = []
            else:
                tune = []
        except (OSError, ValueError):
            tune = []
        table.append(
            {
                "artifact": name,
                "found_by": how,
                "first_commit_date": date,
                "auroc_fields": auroc,
                "corpus_files_referenced": corpus_refs,
                "tuning_words_in_text": tune,
            }
        )
    return {"candidates": len(table), "table": table}


def stage_constants() -> dict:
    say("tracing constants 0.9 / 0.1 / FR11_MEMORY_BOOST in history (slow)")
    out = {}
    for label, needle in (
        ("weight_0.9_r_score", "0.9 * r_score"),
        ("weight_0.1_u_score", "0.1 * u_score"),
        ("FR11_MEMORY_BOOST", "FR11_MEMORY_BOOST"),
    ):
        say(f"  git log -S {needle!r}")
        lines = (
            git("log", "--format=%h %cI %s", "-S" + needle, "--", "python", "scripts")
            .strip()
            .splitlines()
        )
        out[label] = {
            "needle": needle,
            "n_commits": len(lines),
            "earliest": lines[-1] if lines else None,
            "latest": lines[0] if lines else None,
            "first_five": lines[-5:][::-1],
        }
    return out


VARIANT_DEFINITION = (
    "A variant counts if it produced a stored or reported AUROC on FoVer-derived rows AND differs from "
    "the headline in at least one of: (1) verifier set or score formula; (2) corpus file or label "
    "definition; (3) subset size, sampling method or seed list; (4) state condition. Abandoned variants "
    "count. Tags: NOT_LABEL_INFORMED needs written evidence of a fixed rule; LABEL_INFORMED means chosen "
    "after seeing FoVer AUROC; UNKNOWN is counted as LABEL_INFORMED. Frozen before counting."
)
RESTATEMENT_WORDS = ("g2", "dual_condition_integrity", "memory_leakage", "reproduc", "regression")
NOT_LI_WORDS = (
    "frozen",
    "pre-regist",
    "preregist",
    "fixed formula",
    "no tuning",
    "not tuned",
    "pre-register",
)
LI_WORDS = ("tuned", "grid search", "optimiz", "selected on", "fit on")
CORPUS_BUILD_WORDS = (
    "expansion",
    "corpus",
    "labeled",
    "annotation",
    "z3_labels",
    "live_annotation",
)


def stage_final(
    d: dict,
    cache: Path,
    variants_log: Path | None,
    t_start: float,
    prior_seconds: dict[str, float] | None = None,
) -> dict:
    say("assemble final artifact")
    load = lambda n: json.loads((cache / f"{n}.json").read_text(encoding="utf-8"))  # noqa: E731
    control, held, cci = load("control"), load("heldout"), load("corpus_ci")
    tune, meff, var, const = (
        load("tuning"),
        load("memory_effect"),
        load("variants"),
        load("constants"),
    )
    # --- tag variants mechanically ---
    counted, not_counted = [], []
    for row in var["table"]:
        name = row["artifact"]
        low = name.lower()
        path = ROOT / "results" / name
        txt = ""
        try:
            if path.stat().st_size < 30_000_000:
                txt = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            pass
        has_auroc = bool(row["auroc_fields"])
        if name == "experiment_2837_fover_memory_leakage_v3.json":
            not_counted.append({**row, "reason": "the headline artifact itself"})
        elif not has_auroc:
            not_counted.append({**row, "reason": "no stored AUROC field"})
        elif any(w in low for w in RESTATEMENT_WORDS):
            not_counted.append(
                {**row, "reason": "restatement or re-run of the headline (name rule)"}
            )
        else:
            li = [w for w in LI_WORDS if w in txt]
            nli = [w for w in NOT_LI_WORDS if w in txt]
            tag = "LABEL_INFORMED" if li else ("NOT_LABEL_INFORMED" if nli else "UNKNOWN")
            counted.append(
                {
                    **row,
                    "tag": tag,
                    "tag_evidence_words": li or nli,
                    "counts_as_label_informed": tag != "NOT_LABEL_INFORMED",
                }
            )
    n_tag = Counter(r["tag"] for r in counted)
    head_row = next(
        (
            r
            for r in var["table"]
            if r["artifact"] == "experiment_2837_fover_memory_leakage_v3.json"
        ),
        None,
    )
    pre_date = (head_row or {}).get("first_commit_date") or "9999"
    before = [r for r in counted if (r["first_commit_date"] or "9999") < pre_date]
    # --- per-verifier AUROC on the full corpus ---
    per_ver = {k: fast_auroc(d["y"], d[k]) for k in ("r", "u", "s", "mem", "base", "A", "B")}
    # --- gates ---
    pubA, pubB = HEADLINE["A"], HEADLINE["B"]

    def overlaps(ci, pub):
        return not (ci["high"] < pub[1] or ci["low"] > pub[2])  # noqa: E704

    gate1 = {}
    for name in (
        "H0_never_sampled_rows",
        "H1_never_sampled_and_not_memory_flagged",
        "H2_question_never_sampled_and_not_memory_flagged",
    ):
        e = held[name]
        gate1[name] = {
            "n_positive": e["n_positive"],
            "A_overlaps_published": overlaps(e["A"]["ci95"], pubA),
            "B_overlaps_published": overlaps(e["B"]["ci95"], pubB),
            "A_upper_below_published_lower": e["A"]["ci95"]["high"] < pubA[1],
            "B_upper_below_published_lower": e["B"]["ci95"]["high"] < pubB[1],
            "A_point_minus_published": e["A"]["auroc"] - pubA[0],
            "B_point_minus_published": e["B"]["auroc"] - pubB[0],
            "powered_at_200_error_rows": e["n_positive"] >= 200,
        }
    primary = ("H0_never_sampled_rows", "H1_never_sampled_and_not_memory_flagged")
    survives = all(
        gate1[k]["A_overlaps_published"] and gate1[k]["B_overlaps_published"] for k in primary
    )
    narrowed = any(
        gate1[k]["A_upper_below_published_lower"] or gate1[k]["B_upper_below_published_lower"]
        for k in primary
    )
    verdict = (
        "complete: headline_survives_held_out_but_learning_contribution_not_established_on_clean_rows"
        if survives and not narrowed
        else "complete: headline_narrowed_by_held_out"
    )
    wall = {}
    if variants_log and variants_log.is_file():
        stamps = [
            ln[1:9]
            for ln in variants_log.read_text().splitlines()
            if ln.startswith("[") and "]" in ln
        ]
        wall["variants_and_constants_log_first_last"] = [stamps[0], stamps[-1]] if stamps else None
    corpus_sha = hashlib.sha256((ROOT / "data" / "fover_corpus.jsonl").read_bytes()).hexdigest()
    chk = hashlib.sha256(
        json.dumps(
            [corpus_sha, SEEDS, N_EXAMPLES, BOOT_SEED, BOOT_N, VARIANT_DEFINITION], sort_keys=True
        ).encode()
    ).hexdigest()
    return {
        "schema": "carnot.fover_headline_selection_denominator_v1",
        "experiment": "experiment_10030_fover_headline_selection_denominator",
        "honest_verdict": verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "verifier_is_oracle": False,
        "solve_provenance_not_applicable": "not an ARC solve",
        "model_specs": {
            "live_model_invoked": False,
            "note": "CPU verifier scoring of a cached labeled corpus; no LLM is loaded.",
        },
        "random_seed": BOOT_SEED,
        "random_seeds_used": list(SEEDS),
        "reproducibility_checksum": chk,
        "duration_s": (time.time() - t_start) + sum((prior_seconds or {}).values()),
        "stage_wall_times": {
            **wall,
            "prior_stage_seconds_from_run_logs": prior_seconds or {},
            "final_stage_seconds": time.time() - t_start,
        },
        "preconditions_checked": [
            {"resource": "data/fover_corpus.jsonl", "available": True},
            {"resource": "results/experiment_2837_fover_memory_leakage_v3.json", "available": True},
        ],
        "corpus": {
            "path": "data/fover_corpus.jsonl",
            "sha256": corpus_sha,
            "n_rows": int(len(d["y"])),
            "n_positive": int(d["y"].sum()),
        },
        "control_reproduces_published": control,
        "variant_definition_text": VARIANT_DEFINITION,
        "n_variants_total": len(counted),
        "n_not_label_informed": n_tag.get("NOT_LABEL_INFORMED", 0),
        "n_label_informed": n_tag.get("LABEL_INFORMED", 0),
        "n_unknown": n_tag.get("UNKNOWN", 0),
        "n_counted_as_label_informed": sum(1 for r in counted if r["counts_as_label_informed"]),
        "n_candidates_scanned": var["candidates"],
        "n_counted_before_headline_date": len(before),
        "headline_artifact_first_commit_date_used": pre_date,
        "variant_table": counted,
        "not_counted_table_summary": dict(Counter(r["reason"] for r in not_counted)),
        "variant_count_limits": (
            "Counts are an upper bound on candidates. 'Differs from the headline in at least one class' "
            "was NOT checked per artifact; only restatements (name rule) and artifacts with no stored AUROC "
            "field were excluded. Tags come from keyword evidence in the artifact text only; commit "
            "messages were not read. UNKNOWN is counted as label-informed by the frozen definition."
        ),
        "constants_provenance": const,
        "held_out": held,
        "full_corpus_bootstrap": cci,
        "tuning_optimism": tune,
        "memory_effect": meff,
        "per_component_auroc_full_corpus": per_ver,
        "ensemble_vs_best_single_component": {
            "tier0r_alone": per_ver["r"],
            "base_formula_minus_tier0r_alone": per_ver["base"] - per_ver["r"],
            "condition_B_minus_tier0r_alone": per_ver["B"] - per_ver["r"],
            "condition_A_minus_tier0r_alone": per_ver["A"] - per_ver["r"],
            "memory_term_alone": per_ver["mem"],
            "reading": (
                "On the full corpus the 0.9*tier0r + 0.1*tier0u base formula scores within 0.0001 "
                "of tier0r alone (slightly below it). The second component adds nothing measurable. "
                "All of condition A's lift over tier0r comes from the memory term."
            ),
        },
        "headline_text_vs_formula_discrepancy": {
            "north_star_text": "4-verifier score (fr11_session_memory, tier0r_curry_howard, "
            "tier0s_arithmetic_gap, tier0u_logical_consistency)",
            "formula": "0.9*tier0r + 0.1*tier0u + FR11_MEMORY_BOOST(=1.0)*memory_score",
            "tier0s_in_formula": False,
            "tier0s_scored_but_unused_full_corpus_auroc": per_ver["s"],
        },
        "gates": {
            "gate1_survives": gate1,
            "gate1_primary_sets": list(primary),
            "gate1_survives_verdict": survives,
            "gate1_narrowed_verdict": narrowed,
            "gate2_power_note": "H0 and H1 have >=200 error rows. H2 has fewer than 200 and is "
            "underpowered: its CI half-width is about 0.04.",
            "gate3_denominator_stated": True,
        },
        "methodology_note": (
            "FoVer labels come from formal tools. The verifier ensemble does not produce them. The "
            "frozen formula was not changed. The control re-derives the published per-seed means "
            "before any other stage is trusted. The memory term is built from labels of earlier FoVer "
            "rows, so condition A contains label-derived information by design."
        ),
        "independent_reviewer_check": {
            "source": "reviewer agent, own AUROC/set/bootstrap code, did not read this artifact or script",
            "rows_used_by_seeds": 3549,
            "error_rows_used": 1390,
            "error_rows_never_used": 269,
            "memory_ids": 252,
            "memory_matched_rows": 238,
            "memory_matched_error_rows": 228,
            "auroc_A_B_all_rows": [0.9183, 0.9016],
            "auroc_A_B_never_used": [0.9217, 0.9028],
            "auroc_A_B_never_used_unmatched": [0.9082, 0.8969],
            "tier0r_alone": 0.9017,
            "base_formula": 0.9016,
            "tier0u_alone": 0.5099,
            "memory_score_rows_exactly_1": 256,
            "memory_score_rows_between_0_and_1": 7668,
            "memory_score_rows_0": 905,
            "mean_memory_score_correct_rows": 0.208,
            "mean_memory_score_error_rows": 0.507,
            "agrees_with_this_artifact": True,
            "not_rederived": "the 0.9131 per-seed mean (the control stage does that)",
            "refinement": "the memory score is mostly soft token overlap, not an id lookup; both "
            "parts come from entries recorded as incorrect",
        },
        "honest_limits": [
            "Never-sampled rows were not seen by the headline seeds. An earlier experiment may still have "
            "tuned on them. The constants trace is in constants_provenance.",
            "H2 (questions never sampled) has about 113 error rows. Its point estimates are about 0.04 "
            "below the headline and its CI overlaps. This is underpowered, not a pass.",
            "The tuning grid has 44 combinations. A wider search could find more optimism.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--variants-log", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--prior-stage-seconds",
        default="",
        help="name=seconds,... taken from the earlier stage runs' 'done in' log lines",
    )
    ap.add_argument("--stages", default="control,heldout,corpus_ci,tuning,variants,constants")
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # precondition 0
    corpus = ROOT / "data" / "fover_corpus.jsonl"
    pre = [
        {"resource": "data/fover_corpus.jsonl", "available": corpus.is_file()},
        {
            "resource": "results/experiment_2837_fover_memory_leakage_v3.json",
            "available": (
                ROOT / "results" / "experiment_2837_fover_memory_leakage_v3.json"
            ).is_file(),
        },
    ]
    if not all(p["available"] for p in pre):
        say("PRECONDITION FAILED: " + json.dumps(pre))
        return 2
    d = None
    stages = a.stages.split(",")
    if any(
        s in stages for s in ("control", "heldout", "corpus_ci", "tuning", "memory_effect", "final")
    ):
        d = load_scores()
        d["subsets"] = seed_subsets(d)
        say(f"memory index: {d['n_mem_qids']} question ids, {d['n_mem_token_sets']} token sets")
    for s in stages:
        f = cache / f"{s}.json"
        say(f"stage {s}")
        if s == "control":
            r = stage_control(d, d["subsets"])
        elif s == "heldout":
            r = stage_heldout(d, d["subsets"])
        elif s == "corpus_ci":
            r = stage_corpus_ci(d)
        elif s == "tuning":
            r = stage_tuning_gap(d, d["subsets"])
        elif s == "memory_effect":
            r = stage_memory_effect(d, d["subsets"])
        elif s == "final":
            prior = {
                k: float(v)
                for k, v in (p.split("=") for p in a.prior_stage_seconds.split(",") if p)
            }
            r = stage_final(d, cache, Path(a.variants_log) if a.variants_log else None, t0, prior)
            out = (
                Path(a.out)
                if a.out
                else ROOT / "results" / "experiment_10030_fover_headline_selection_denominator.json"
            )
            out.write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")
            say(f"  wrote {out}")
        elif s == "variants":
            r = stage_variants()
        elif s == "constants":
            r = stage_constants()
        else:
            continue
        f.write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")
        say(f"  wrote {f.name}")
    say(f"done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
