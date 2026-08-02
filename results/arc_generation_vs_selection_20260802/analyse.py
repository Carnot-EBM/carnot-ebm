#!/usr/bin/env python3
"""Analyse the held-out census. Descriptive only -- no treatment, no pairing, no p-value."""

from __future__ import annotations

import glob
import json
import pathlib
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "out"

GATES = [
    ("plain_exact", "accuracy", 0.5),
    ("plain_cell_recall", "cell_recall", 0.5),
    ("hidden_state_trust", "heldout_change_consistency", 0.5),
    ("change_gate_fidelity", "change_fidelity", 0.5),
    ("change_accuracy_deinfl", "change_accuracy", 0.5),
]


def q(vals, p):
    v = sorted(vals)
    return v[min(len(v) - 1, int(p * (len(v) - 1)))]


def dist(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    return {
        "n": len(v),
        "min": round(v[0], 4),
        "q1": round(q(v, 0.25), 4),
        "median": round(q(v, 0.5), 4),
        "q3": round(q(v, 0.75), 4),
        "max": round(v[-1], 4),
    }


def rule_of_three(n):
    """One-sided 95% upper bound on a rate when 0 of n were observed."""
    return round(1 - 0.05 ** (1 / n), 5) if n else None


def load_rows():
    rows, meta = [], {}
    for f in sorted(glob.glob(str(OUT / "cells" / "*.json"))):
        d = json.loads(pathlib.Path(f).read_text())
        meta[d["game"]] = {
            k: d.get(k) for k in ("hidden_state_branch", "splits", "window_error", "fresh_error")
        }
        for r in d.get("rows", []):
            r["game"] = d["game"]
            r["hidden_state_branch"] = d.get("hidden_state_branch")
            rows.append(r)
    for f in sorted(glob.glob(str(OUT / "cells_bestofn" / "*.json"))):
        d = json.loads(pathlib.Path(f).read_text())
        if d.get("status") != "ok":
            meta.setdefault(d["game"], {})["bestofn"] = d.get("status")
            continue
        meta.setdefault(d["game"], {})["bestofn_splits"] = d.get("splits")
        meta[d["game"]]["bestofn_split_check"] = d.get("split_check")
        for r in d.get("rows", []):
            r["game"] = d["game"]
            r["hidden_state_branch"] = d.get("hidden_state_branch")
            rows.append(r)
    return rows, meta


def main() -> int:
    rows, meta = load_rows()
    real = [r for r in rows if r["corpus"] not in ("control",)]
    ctrl = [r for r in rows if r["corpus"] == "control"]
    ok = [r for r in real if r.get("status") == "ok"]

    report = {"n_rows": len(rows), "n_real_rows": len(real), "n_ok_rows": len(ok)}

    # ---- INSTRUMENT CHECK + THE STATELESS CEILING. ----
    # The oracle is a LOOKUP on (grid, action). The engines have signature
    # (grid, action, data) and carry no state between calls, so the oracle IS THE CEILING
    # for the entire engine class. Where it scores < 1.0 that is NOT a broken instrument:
    # it is a game where (grid, action) does not determine next_grid -- hidden state -- and
    # NO stateless engine can do better. Reported per game as a ceiling, and used as the
    # real instrument assertion: no engine may EXCEED its own game/split oracle.
    orc = [r for r in ctrl if r["cell"] == "__control_oracle__" and r.get("status") == "ok"]
    idn = [r for r in ctrl if r["cell"] == "__control_identity__" and r.get("status") == "ok"]
    ceil = {(r["game"], r["split"]): r for r in orc}
    violations = []
    for r in ok:
        c = ceil.get((r["game"], r["split"]))
        if c and (r.get("accuracy") or 0) > (c.get("accuracy") or 0) + 1e-9:
            violations.append(
                f"{r['cell']}/{r['split']}: {r.get('accuracy')} > oracle {c.get('accuracy')}"
            )
    report["stateless_ceiling"] = {
        "what": (
            "oracle = lookup on (grid, action). Engines are stateless functions of "
            "(grid, action, data), so this is the CEILING for the engine class, not "
            "merely a plumbing check. oracle < 1.0 means the game has hidden state and "
            "no stateless engine can reach 1.0 there."
        ),
        "per_game_split": {
            f"{g}/{s}": {
                "oracle_accuracy": r.get("accuracy"),
                "oracle_cell_recall": r.get("cell_recall"),
                "oracle_change_accuracy": r.get("change_accuracy"),
            }
            for (g, s), r in sorted(ceil.items())
        },
        "n_game_splits_where_ceiling_is_1.0": sum(
            1 for r in orc if (r.get("accuracy") or 0) >= 1.0
        ),
        "n_game_splits_where_ceiling_below_1.0": sum(
            1 for r in orc if (r.get("accuracy") or 0) < 1.0
        ),
        "ceiling_below_1_means_hidden_state": sorted(
            f"{g}/{s}={round(r.get('accuracy') or 0, 3)}"
            for (g, s), r in ceil.items()
            if (r.get("accuracy") or 0) < 1.0
        ),
    }
    report["instrument_check"] = {
        "oracle_cells": len(orc),
        "no_engine_exceeds_its_own_oracle": not violations,
        "ceiling_violations": violations[:20],
        "oracle_reaches_1.0_on_at_least_one_split_per_metric": any(
            (r.get("accuracy") or 0) >= 1.0 for r in orc
        ),
        "identity_cells": len(idn),
        "identity_accuracy": dist([r.get("accuracy") for r in idn]),
        "identity_change_accuracy": dist([r.get("change_accuracy") for r in idn]),
        "identity_cell_recall": dist([r.get("cell_recall") for r in idn]),
        "identity_heldout_change_consistency": dist(
            [r.get("heldout_change_consistency") for r in idn]
        ),
        "identity_trust_pass_count": sum(1 for r in idn if r.get("hidden_state_trust_pass")),
        "reading": (
            "the oracle proves the metric plumbing can register a full pass; the "
            "identity row is what a degenerate engine scores, per split, and is the "
            "reference any 'high' induced score must beat."
        ),
    }

    report["statuses"] = dict(Counter(r.get("status") for r in real))
    report["excluded_not_zeroed"] = {
        s: c for s, c in Counter(r.get("status") for r in real).items() if s != "ok"
    }
    report["per_game_split_sizes"] = meta

    # ---- distributions + clearance, per split, pooled and per corpus ----
    per_split = {}
    for split in sorted({r["split"] for r in ok if r.get("split")}):
        allrows_dup = [r for r in ok if r["split"] == split]
        # DE-DUPLICATE BY ENGINE BODY. Two cells whose engine sha256 is identical are ONE
        # observation, not two: the inert A/B emits an `off` and an `on` cell per
        # (game, replicate) and where the treatment did not fire both carry the same text.
        _seen, allrows = set(), []
        for _r in sorted(allrows_dup, key=lambda x: x["cell"]):
            k = (_r["game"], _r.get("engine_sha256"))
            if _r.get("engine_sha256") and k in _seen:
                continue
            _seen.add(k)
            allrows.append(_r)
        # vr.n == 0 means WorldModelVerifier graded no row at all (e.g. a tail that is the
        # level-up row only, which it excludes by design). That is VACUOUS, not a zero.
        srows = [r for r in allrows if (r.get("n_scored") or 0) > 0]
        # a row whose held-out slice contains no gradable CHANGING transition cannot
        # discriminate an identity engine from a correct one: excluded from the change-family
        # TWO gradability notions, because the two shipped gates disagree. WorldModelVerifier
        # `continue`s on a level-up row BEFORE counting it (the completing action re-lays the
        # playfield out), so vr_n_changing can be 0 where the pooled consistency function --
        # which applies no such exclusion -- still sees changed cells. A row the verifier
        # grades nothing on is UNGRADABLE, not a zero, and is excluded from the verifier
        # family rather than averaged in as a failure.
        gradable = [r for r in srows if (r.get("vr_n_changing") or 0) > 0]
        cw_gradable = [r for r in srows if (r.get("cw_true_changed_cells") or 0) > 0]
        block = {
            "n_engine_rows": len(srows),
            "n_rows_before_engine_dedup": len(allrows_dup),
            "n_rows_dropped_as_byte_identical_duplicates": len(allrows_dup) - len(allrows),
            "n_rows_dropped_vacuous_verifier_graded_nothing": len(allrows) - len(srows),
            "n_rows_with_a_gradable_change": len(gradable),
            "n_scored_transitions": dist([r.get("n_scored") for r in srows]),
        }
        block["gradability_disagreement"] = {
            "verifier_gradable": len(gradable),
            "consistency_gradable": len(cw_gradable),
            "note": (
                "WorldModelVerifier excludes level-up rows; "
                "score_change_weighted_consistency does not. Where these differ the two "
                "shipped gates are scoring different row sets."
            ),
        }
        for gname, field, thr in GATES:
            pool = (
                srows
                if gname == "plain_exact"
                else cw_gradable
                if gname == "hidden_state_trust"
                else gradable
            )
            vals = [r.get(field) for r in pool if r.get(field) is not None]
            clears = [r for r in pool if (r.get(field) or 0) >= thr]
            games_any = sorted({r["game"] for r in clears})
            block[gname] = {
                "threshold": thr,
                "distribution": dist(vals),
                "n_clearing": len(clears),
                "n_pool": len(pool),
                "pct_clearing": round(100 * len(clears) / len(pool), 2) if pool else None,
                "n_distinct_games_with_any_clear": len(games_any),
                "games_with_any_clear": games_any,
                "rule_of_three_upper_bound_if_zero": rule_of_three(len(pool))
                if not clears
                else None,
            }
        # the two gates as the code actually decides them (conjunctions, not bare thresholds)
        tp = [r for r in cw_gradable if r.get("hidden_state_trust_pass")]
        cg = [r for r in gradable if r.get("change_gate_pass")]
        r10 = [r for r in srows if r.get("refinement_loop_pass_1p0")]
        block["shipped_conjunctions"] = {
            "hidden_state_trust_pass (consistency>=0.5 AND correct_changed_cells>=1)": {
                "n": len(tp),
                "of": len(cw_gradable),
                "games": sorted({r["game"] for r in tp}),
            },
            "change_gate_pass (default-OFF)": {
                "n": len(cg),
                "of": len(gradable),
                "games": sorted({r["game"] for r in cg}),
            },
            "refinement_loop min_heldout_accuracy=1.0 (live hardcoded)": {
                "n": len(r10),
                "of": len(srows),
                "games": sorted({r["game"] for r in r10}),
            },
            "nondegeneracy alone (correct_changed_cells>=1)": {
                "n": sum(1 for r in cw_gradable if r.get("cw_nondegenerate")),
                "of": len(cw_gradable),
            },
        }
        # precision decomposition for the recall-passers: scribbler or predictor?
        rp = [r for r in gradable if (r.get("cell_recall") or 0) >= 0.5]
        block["recall_passers_precision"] = {
            "n": len(rp),
            "precision_distribution": dist([r.get("precision") for r in rp]),
            "n_also_change_accuracy_ge_0.5": sum(
                1 for r in rp if (r.get("change_accuracy") or 0) >= 0.5
            ),
            "n_with_precision_ge_0.9": sum(1 for r in rp if (r.get("precision") or 0) >= 0.9),
            "games": sorted({r["game"] for r in rp}),
        }
        block["by_corpus"] = {}
        for c in sorted({r["corpus"] for r in srows}):
            cr = [r for r in gradable if r["corpus"] == c]
            block["by_corpus"][c] = {
                "n": len(cr),
                "cell_recall": dist([r.get("cell_recall") for r in cr]),
                "heldout_change_consistency": dist(
                    [r.get("heldout_change_consistency") for r in cr]
                ),
                "change_accuracy": dist([r.get("change_accuracy") for r in cr]),
                "n_trust_pass": sum(1 for r in cr if r.get("hidden_state_trust_pass")),
            }
        # per-game maxima -- the roster view, so no pooled number can hide behind one game
        bg = defaultdict(list)
        for r in gradable:
            bg[r["game"]].append(r)
        block["per_game"] = {
            g: {
                "n": len(v),
                "max_change_accuracy": round(max((r.get("change_accuracy") or 0) for r in v), 4),
                "max_cell_recall": round(max((r.get("cell_recall") or 0) for r in v), 4),
                "max_heldout_change_consistency": round(
                    max((r.get("heldout_change_consistency") or 0) for r in v), 4
                ),
                "max_change_fidelity": round(max((r.get("change_fidelity") or 0) for r in v), 4),
                "best_precision_at_recall_ge_0.5": (
                    round(
                        max(
                            (r.get("precision") or 0)
                            for r in v
                            if (r.get("cell_recall") or 0) >= 0.5
                        ),
                        4,
                    )
                    if any((r.get("cell_recall") or 0) >= 0.5 for r in v)
                    else None
                ),
                "n_trust_pass": sum(1 for r in v if r.get("hidden_state_trust_pass")),
                "hidden_state_branch": v[0].get("hidden_state_branch"),
            }
            for g, v in sorted(bg.items())
        }
        per_split[split] = block
    report["per_split"] = per_split

    (OUT / "analysis.json").write_text(json.dumps(report, indent=1, default=str))
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("n_rows", "n_real_rows", "n_ok_rows", "instrument_check", "statuses")
            },
            indent=1,
            default=str,
        )[:2500]
    )
    for s, b in per_split.items():
        print(
            f"\n===== SPLIT {s} =====  rows={b['n_engine_rows']} "
            f"gradable={b['n_rows_with_a_gradable_change']}"
        )
        for gname, _f, thr in GATES:
            g = b[gname]
            print(
                f"  {gname:24} >= {thr}: {g['n_clearing']:>3}/{g['n_pool']:<3} "
                f"({g['pct_clearing']}%) games={g['n_distinct_games_with_any_clear']:>2} "
                f"dist={g['distribution']}"
            )
        print("  conjunctions:", json.dumps(b["shipped_conjunctions"], default=str)[:400])
        print(
            "  recall-passer precision:",
            json.dumps(b["recall_passers_precision"], default=str)[:300],
        )
    print("\n-> out/analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
