#!/usr/bin/env python3
"""Pull the decision-relevant numbers out of analysis.json + the live corpus, side by side.

Console-formatting only -- this file prints the fixed-width tables that were read while
drawing the conclusion. It computes nothing the artifact depends on; build_artifact.py is the
one that writes results. E501 is disabled for the file because its long lines are f-string
table rows whose alignment IS the readability, and hand-wrapping them would make the tables
harder to read to satisfy a width rule.
"""
# ruff: noqa: E501

from __future__ import annotations

import glob
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
A = json.loads(pathlib.Path(HERE / "out" / "analysis.json").read_text())


def rows():
    out = []
    for f in sorted(glob.glob(str(HERE / "out" / "cells" / "*.json"))) + sorted(
        glob.glob(str(HERE / "out" / "cells_bestofn" / "*.json"))
    ):
        d = json.loads(pathlib.Path(f).read_text())
        for r in d.get("rows", []):
            r["game"] = d["game"]
            r["hs"] = d.get("hidden_state_branch")
            out.append(r)
    return out


def live():
    recs = []

    def walk(o, g, arm):
        if isinstance(o, dict):
            for a in o.get("induction_attempt_gate_diagnostics") or []:
                recs.append({**a, "_game": g, "_arm": arm})
            for v in o.values():
                walk(v, g, arm)
        elif isinstance(o, list):
            for v in o:
                walk(v, g, arm)

    for f in sorted(glob.glob(str(REPO / "results/first_win_llm_on_20260727/cells/*.json"))):
        d = json.loads(pathlib.Path(f).read_text())
        walk(d, d.get("game"), d.get("arm"))
    return recs


def r3(n):
    return round(1 - 0.05 ** (1 / n), 4) if n else None


R = [r for r in rows() if r.get("status") == "ok" and r["corpus"] != "control"]
L = live()
hs = [r for r in L if r.get("heldout_change_consistency") is not None]
pl = [r for r in L if r.get("verify_accuracy") is not None]

print("=" * 100)
print(
    "HARNESS (offline, curated winning-trajectory window)  vs  LIVE (the actual 2026-07-27 rejection set)"
)
print("=" * 100)
for split in ("A_tail", "P_proven_heldout", "C_fresh120"):
    b = A["per_split"].get(split)
    if not b:
        continue
    print(
        f"\n--- SPLIT {split}  (rows={b['n_engine_rows']}, "
        f"dropped-vacuous={b.get('n_rows_dropped_vacuous_verifier_graded_nothing')}) ---"
    )
    for k in (
        "plain_exact",
        "plain_cell_recall",
        "hidden_state_trust",
        "change_accuracy_deinfl",
        "change_gate_fidelity",
    ):
        g = b[k]
        d = g["distribution"]
        z = (
            f" [rule-of-3 upper bound {g['rule_of_three_upper_bound_if_zero']}]"
            if g["n_clearing"] == 0
            else ""
        )
        print(
            f"  {k:24} {g['n_clearing']:>3}/{g['n_pool']:<4} games={g['n_distinct_games_with_any_clear']:>2}  "
            f"med={d['median']:.4f} q3={d['q3']:.4f} max={d['max']:.4f}{z}"
        )
    rp = b["recall_passers_precision"]
    print(
        f"  recall-passers n={rp['n']} precision med={(rp['precision_distribution'] or {}).get('median')} "
        f"| n_precision>=0.9 = {rp['n_with_precision_ge_0.9']} "
        f"| n_also_change_accuracy>=0.5 = {rp['n_also_change_accuracy_ge_0.5']}"
    )

print(
    "\n--- LIVE, the 52 of 136 attempts that recorded a margin (84 have NO margin: missing, not zero) ---"
)


def d(v):
    v = sorted(x for x in v if x is not None)

    def q(p):
        return v[min(len(v) - 1, int(p * (len(v) - 1)))]

    return f"n={len(v):>3} med={q(0.5):.4f} q3={q(0.75):.4f} max={v[-1]:.4f}"


print(
    f"  hidden_state consistency  {d([r['heldout_change_consistency'] for r in hs])}  "
    f">=0.5: {sum(1 for r in hs if r['heldout_change_consistency'] >= 0.5)}/{len(hs)} "
    f"[rule-of-3 {r3(len(hs))}]"
)
print(
    f"  correct_changed_cells==0  {sum(1 for r in hs if (r.get('correct_changed_cells') or 0) == 0)}/{len(hs)}"
    "  (fails nondegeneracy at ANY threshold)"
)
print(
    f"  plain verify_accuracy     {d([r['verify_accuracy'] for r in pl])}  "
    f">=0.5: {sum(1 for r in pl if r['verify_accuracy'] >= 0.5)}/{len(pl)}"
)
print(
    f"  plain verify_cell_recall  {d([r['verify_cell_recall'] for r in pl])}  "
    f">=0.5: {sum(1 for r in pl if r['verify_cell_recall'] >= 0.5)}/{len(pl)} "
    f"[rule-of-3 {r3(len(pl))}]"
)

# the SELECTION case, stated as strongly as the data allows, then tested
print("\n" + "=" * 100)
print(
    "THE STRONGEST SELECTION CASE: engines that are high-recall AND high-precision on a never-fitted split"
)
print("=" * 100)
for split in ("A_tail", "P_proven_heldout", "C_fresh120"):
    sel = [
        r
        for r in R
        if r["split"] == split
        and (r.get("cell_recall") or 0) >= 0.5
        and (r.get("precision") or 0) >= 0.9
        and (r.get("vr_n_changing") or 0) > 0
    ]
    gs = sorted({r["game"] for r in sel})
    print(f"  {split:18} n={len(sel):>3} games={len(gs):>2} {gs}")
    if sel:
        ca = sorted(r.get("change_accuracy") or 0 for r in sel)
        print(
            f"{'':21} their change_accuracy (WHOLE grid right): med={ca[len(ca) // 2]:.4f} max={ca[-1]:.4f}"
            f" | n>=0.5: {sum(1 for x in ca if x >= 0.5)}"
        )

print(
    "\n--- per-game max, SPLIT A_tail (roster view; no pooled number may hide behind one game) ---"
)
pg = A["per_split"]["A_tail"]["per_game"]
print(
    f"{'game':6} {'hs':>3} {'n':>3} {'max_chacc':>9} {'max_cellrec':>11} {'max_hcc':>8} {'max_chfid':>9} {'trustpass':>9}"
)
for g, v in pg.items():
    print(
        f"{g:6} {str(v['hidden_state_branch'])[0]:>3} {v['n']:>3} {v['max_change_accuracy']:>9.4f} "
        f"{v['max_cell_recall']:>11.4f} {v['max_heldout_change_consistency']:>8.4f} "
        f"{v['max_change_fidelity']:>9.4f} {v['n_trust_pass']:>9}"
    )
n_zero = sum(1 for v in pg.values() if v["max_change_accuracy"] == 0)
print(f"\ngames whose BEST engine never predicts one whole changing grid: {n_zero} of {len(pg)}")
