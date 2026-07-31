#!/usr/bin/env python3
"""PHASE 1 (confirm), FINAL STEP -- score every completion OUT-OF-SAMPLE and compute the verdict.

WHAT IS NEW HERE, and it is the entire reason this phase exists. Every previous scoring of this
induce path was IN-SAMPLE: `results/arc_engine_validation_20260731/gate_scores.json` says so in
its own payload, because every transition it graded on had been rendered into the induce prompt.
That is what makes Phase 1's `cell_recall 0.947` uncitable -- all six of the changing transitions
it was graded on were shown to the model. This scorer grades on the split PROVEN by `split.py`:
rows of the agent's collected corpus whose rendered delta line does not occur in the prompt text.

THE SPLIT IS STATED, NOT ASSUMED. Per game the report carries `n_shown`, `n_heldout`,
`heldout_n_changing` and `heldout_n_noop`, so a reader can see exactly what was graded. Two of the
six games have a held-out set containing ZERO grid-changing transitions, and one has an EMPTY
held-out set; those are properties of the live corpus at the real induction point, and they are
reported as unscoreable rather than quietly scored.

WHY BOTH `usable` AND QUALITY, ALWAYS TOGETHER. `usable` (accepted + mechanically clean + changes
the grid somewhere) is necessary and nowhere near sufficient: Phase 2 produced a tu93 engine that
cleared it with `cell_recall 0.112`, 0 of 25 changes correct and 144 INVENTED cells. It changed
the grid wrongly. So every usable verdict here is reported next to the numbers that say whether
the change was right, and the funnel is counted at BOTH bars.

WHY `heldout_cell_recall` IS None RATHER THAN 0.0 WHEN THERE ARE NO CHANGING ROWS. A held-out set
of pure no-ops cannot distinguish a good engine from the identity function -- the identity scores
a perfect held-out accuracy on it. Writing 0.0 would look like a measured failure and writing 1.0
would look like a measured success; both are lies about a quantity that was not measured. None is
the truth, and the paired tests skip those cells rather than counting them either way.

NO THRESHOLD IS CHANGED ANYWHERE. `min_heldout_accuracy`, the degenerate-goal predicate and the
root-goal predicate are untouched; the 0.5 recall line used to sort "models something" from
"models nothing" is a REPORTING bar for this note, applied to numbers the shipped verifier
produced, and it is stated as such rather than written into any gate.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import time

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "confirm_scored.json"

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_confirm_score/e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, str(HERE))

TAGS = [t for t in os.environ.get("SCORE_TAGS", "gpu0,gpu1").split(",") if t]
# REPORTING bar for this note (see module docstring). Not a gate, not written anywhere.
QUALITY_RECALL_BAR = 0.5


def _sign_test(n_a: int, n_b: int) -> float:
    """Exact two-sided sign test on discordant pairs. n_a wins for A, n_b for B."""

    n = n_a + n_b
    if n == 0:
        return 1.0
    k = min(n_a, n_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def main() -> int:  # noqa: C901
    import numpy as np

    from carnot.agentic import arc_executable_world_model as e3
    from split import load_split

    scored: dict = {}  # (game, tag) -> metrics
    attempts: list[dict] = []
    splits: dict = {}
    witnesses: dict = {}
    run_status: dict = {}

    for tag in TAGS:
        path = HERE / "confirm" / tag / "confirm.json"
        if not path.exists():
            run_status[tag] = "NOT_RUN"
            continue
        data = json.loads(path.read_text())
        run_status[tag] = data.get("status")
        witnesses[tag] = data.get("witness")
        for game in data.get("games", []):
            if game not in splits:
                splits[game] = load_split(game, data.get("call_index", 2))

        for row in data.get("rows", []):
            if row.get("status") != "ok":
                continue
            game = row["game"]
            s = splits[game]
            text = (HERE / "confirm" / tag / row["completion_file"]).read_text(errors="replace")
            code = e3._extract_python(text) or text.strip()
            key = (game, row["tag"])
            m: dict = {
                "game": game,
                "tag": row["tag"],
                "usable": row["usable"],
                "defect_kinds": row["defect_kinds"],
                "stop_type": row["stop_type"],
                "predicted_n": row["predicted_n"],
                "wall_s": row["wall_s"],
                "engine_changes_anything": row["engine_changes_anything"],
            }
            ns: dict = {"np": np, "numpy": np}
            try:
                exec(compile(code, row["completion_file"], "exec"), ns)  # noqa: S102
                engine = ns.get("engine")
                assert callable(engine)
            except Exception as exc:  # noqa: BLE001
                m["score_status"] = f"unrunnable:{type(exc).__name__}"
                scored[key] = m
                continue
            m["score_status"] = "ok"

            for label, rows_ in (("heldout", s["_heldout"]), ("in_sample", s["_shown"])):
                if not rows_:
                    m[f"{label}_n"] = 0
                    continue
                vr = e3.WorldModelVerifier(list(rows_)).score(engine)
                n_changing = int(getattr(vr, "n_changing", 0) or 0)
                m[f"{label}_n"] = int(getattr(vr, "n", 0) or 0)
                m[f"{label}_accuracy"] = round(float(getattr(vr, "accuracy", 0.0) or 0.0), 4)
                # None, not 0.0: with no changing rows this quantity was NOT measured.
                m[f"{label}_cell_recall"] = (
                    round(float(getattr(vr, "cell_recall", 0.0) or 0.0), 4)
                    if n_changing
                    else None
                )
                m[f"{label}_change_fidelity"] = (
                    round(float(getattr(vr, "change_fidelity", 0.0) or 0.0), 4)
                    if n_changing
                    else None
                )
                m[f"{label}_n_changing"] = n_changing
                m[f"{label}_n_changes_correct"] = int(getattr(vr, "n_changes_correct", 0) or 0)
                m[f"{label}_invented_changed_cells"] = int(
                    getattr(vr, "invented_changed_cells", 0) or 0
                )
                m[f"{label}_n_noop"] = int(getattr(vr, "n_noop", 0) or 0)
                m[f"{label}_n_noop_hallucinated"] = int(
                    getattr(vr, "n_noop_hallucinated", 0) or 0
                )
            scored[key] = m

        for rec in data.get("attempts", []):
            out = {
                "game": rec["game"],
                "seed_base": rec["seed_base"],
                "attempt": rec["attempt"],
                "seed": rec["seed"],
                "temperature": rec["temperature"],
                "reask_fired": rec.get("reask_fired"),
            }
            for arm in ("CONTROL", "TREATMENT_round0", "TREATMENT_final"):
                r = rec.get(arm)
                if isinstance(r, dict) and r.get("status") == "ok":
                    out[arm] = scored.get((rec["game"], r["tag"]))
                else:
                    out[arm] = None
            attempts.append(out)

    # ---- per-game funnel, at BOTH bars -------------------------------------------------
    def _quality_ok(m: dict | None) -> bool | None:
        """None = the held-out set cannot grade change quality on this game."""
        if not m or not m.get("usable"):
            return False
        if m.get("heldout_cell_recall") is None:
            return None
        return bool(m["heldout_cell_recall"] > QUALITY_RECALL_BAR)

    games = sorted({a["game"] for a in attempts})
    per_game = []
    for game in games:
        rows_ = [a for a in attempts if a["game"] == game]
        s = splits[game]
        g = {
            "game": game,
            "n_attempts": len(rows_),
            "n_shown": s["n_shown"],
            "n_heldout": s["n_heldout"],
            "heldout_n_changing": s["heldout_n_changing"],
            "heldout_n_noop": s["heldout_n_noop"],
            "heldout_can_grade_change": s["heldout_can_grade_change"],
        }
        for arm in ("CONTROL", "TREATMENT_round0", "TREATMENT_final"):
            ms = [a[arm] for a in rows_ if a[arm]]
            usable = [m for m in ms if m.get("usable")]
            g[f"{arm}_n_usable"] = len(usable)
            g[f"{arm}_any_usable"] = bool(usable)
            recalls = [
                m["heldout_cell_recall"] for m in usable if m.get("heldout_cell_recall") is not None
            ]
            g[f"{arm}_best_heldout_cell_recall"] = max(recalls) if recalls else None
            best = None
            for m in usable:
                if m.get("heldout_cell_recall") is None:
                    continue
                if best is None or m["heldout_cell_recall"] > best["heldout_cell_recall"]:
                    best = m
            g[f"{arm}_best"] = (
                {
                    k: best[k]
                    for k in (
                        "tag",
                        "heldout_n",
                        "heldout_accuracy",
                        "heldout_cell_recall",
                        "heldout_change_fidelity",
                        "heldout_n_changing",
                        "heldout_n_changes_correct",
                        "heldout_invented_changed_cells",
                        "heldout_n_noop_hallucinated",
                        "in_sample_cell_recall",
                    )
                }
                if best
                else None
            )
            q = [_quality_ok(m) for m in ms]
            g[f"{arm}_any_quality"] = (
                True if any(x is True for x in q) else (None if any(x is None for x in q) else False)
            )
        per_game.append(g)

    # ---- paired tests, attempt-matched ------------------------------------------------
    def paired(arm_a: str, arm_b: str, pred) -> dict:
        a_only = b_only = both = neither = 0
        for rec in attempts:
            ma, mb = rec[arm_a], rec[arm_b]
            if ma is None or mb is None:
                continue
            pa, pb = pred(ma), pred(mb)
            if pa is None or pb is None:
                continue
            if pa and not pb:
                a_only += 1
            elif pb and not pa:
                b_only += 1
            elif pa and pb:
                both += 1
            else:
                neither += 1
        return {
            "arm_a": arm_a,
            "arm_b": arm_b,
            f"{arm_a}_only": a_only,
            f"{arm_b}_only": b_only,
            "both": both,
            "neither": neither,
            "n_discordant": a_only + b_only,
            "sign_test_p_two_sided": round(_sign_test(a_only, b_only), 4),
        }

    def _is_usable(m: dict) -> bool:
        return bool(m.get("usable"))

    def _is_quality(m: dict):
        if not m.get("usable"):
            return False
        if m.get("heldout_cell_recall") is None:
            return None
        return bool(m["heldout_cell_recall"] > QUALITY_RECALL_BAR)

    tests = {
        "usable__treatment_final_vs_control": paired("TREATMENT_final", "CONTROL", _is_usable),
        "usable__treatment_round0_vs_control": paired("TREATMENT_round0", "CONTROL", _is_usable),
        "usable__treatment_final_vs_round0": paired(
            "TREATMENT_final", "TREATMENT_round0", _is_usable
        ),
        "quality__treatment_final_vs_control": paired("TREATMENT_final", "CONTROL", _is_quality),
    }

    n_games = len(games)
    funnel = {
        arm: {
            "games_any_usable": sum(1 for g in per_game if g[f"{arm}_any_usable"]),
            "games_any_quality": sum(1 for g in per_game if g[f"{arm}_any_quality"] is True),
            "games_quality_unscoreable": sum(
                1 for g in per_game if g[f"{arm}_any_quality"] is None
            ),
            "n_games": n_games,
        }
        for arm in ("CONTROL", "TREATMENT_round0", "TREATMENT_final")
    }

    # shipped accept-first behaviour, restated on this run's data
    ctl_rows = [a["CONTROL"] for a in attempts if a["CONTROL"]]
    accept_defective = sum(1 for m in ctl_rows if m["defect_kinds"])
    payload = {
        "generated_by": "results/arc_induce_confirm_20260731/harness/score.py",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_status": run_status,
        "witnesses": witnesses,
        "OUT_OF_SAMPLE_SPLIT": (
            "HELD_OUT = the agent's collected transitions MINUS the <=8 rows _transitions_block "
            "renders into the induce prompt, proven per game in split.json by checking each "
            "row's rendered delta line against the prompt text. `heldout_cell_recall` is null "
            "where the held-out set contains no grid-changing row -- that quantity was not "
            "measured, and an identity engine scores a perfect held-out accuracy there."
        ),
        "quality_recall_bar": QUALITY_RECALL_BAR,
        "quality_bar_note": (
            "REPORTING bar for this note only. No production threshold was read, changed or "
            "written by this run."
        ),
        "n_attempts_total": len(attempts),
        "control_accepted_a_defective_candidate": accept_defective,
        "control_attempts_scored": len(ctl_rows),
        "funnel": funnel,
        "paired_tests": tests,
        "per_game": per_game,
        "attempts": attempts,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"run_status: {run_status}   attempts={len(attempts)}")
    print(
        f"shipped CONTROL accepted a MECHANICALLY DEFECTIVE candidate in "
        f"{accept_defective}/{len(ctl_rows)} attempts"
    )
    print("\nper game (best USABLE engine per arm, scored OUT-OF-SAMPLE):")
    hdr = (
        f"{'game':5s} {'heldout':>16s}  {'arm':16s} {'usable':>7s} {'recall':>7s} "
        f"{'fidel':>6s} {'chg ok':>8s} {'invent':>7s} {'noop_h':>7s}"
    )
    print(hdr)
    for g in per_game:
        ho = f"n={g['n_heldout']} chg={g['heldout_n_changing']}"
        for arm in ("CONTROL", "TREATMENT_round0", "TREATMENT_final"):
            b = g[f"{arm}_best"]
            n_us = g[f"{arm}_n_usable"]
            if b is None:
                print(
                    f"{g['game']:5s} {ho:>16s}  {arm:16s} {n_us:>3d}/{g['n_attempts']:<3d} "
                    f"{'--':>7s} {'--':>6s} {'--':>8s} {'--':>7s} {'--':>7s}"
                )
                continue
            print(
                f"{g['game']:5s} {ho:>16s}  {arm:16s} {n_us:>3d}/{g['n_attempts']:<3d} "
                f"{b['heldout_cell_recall']:>7.3f} {b['heldout_change_fidelity']:>6.3f} "
                f"{b['heldout_n_changes_correct']:>3d}/{b['heldout_n_changing']:<4d} "
                f"{b['heldout_invented_changed_cells']:>7d} "
                f"{b['heldout_n_noop_hallucinated']:>7d}"
            )
    print("\nfunnel:")
    for arm, f in funnel.items():
        print(
            f"  {arm:16s} usable {f['games_any_usable']}/{f['n_games']} games   "
            f"quality>{QUALITY_RECALL_BAR} {f['games_any_quality']}/{f['n_games']} "
            f"({f['games_quality_unscoreable']} unscoreable out-of-sample)"
        )
    print("\npaired tests (attempt-matched):")
    for name, t in tests.items():
        print(
            f"  {name:42s} {t['arm_a']}_only={t[t['arm_a'] + '_only']} "
            f"{t['arm_b']}_only={t[t['arm_b'] + '_only']} both={t['both']} "
            f"neither={t['neither']}  p={t['sign_test_p_two_sided']}"
        )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
