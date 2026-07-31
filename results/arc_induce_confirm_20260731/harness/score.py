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

BUT A NO-OP-ONLY HELD-OUT SET IS NOT A WORTHLESS ONE, and the reason is specific to these games.
ft09's 25 collected transitions are ALL ACTION6 clicks: 6 change 38 cells each and 19 change
nothing. lp85's are all clicks too -- 2 change 293 cells, 23 change nothing. So on those two games
the mechanic IS target discrimination, and the held-out no-ops ask the load-bearing question:
does the engine know WHERE the click has to land, or does it fire its induced rule at every
click? An engine that hallucinates a change on all 16 unseen inert clicks has not learned the
mechanic, and that verdict is only visible out-of-sample -- in-sample it was shown 6 hits and 2
misses and can reproduce them from position alone. What the no-op channel still cannot do is
separate a correct engine from the identity function, which is why `heldout_n_noop_hallucinated`
is always reported beside `engine_changes_anything` on the shown rows.

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
    def _quality_strict(m: dict | None) -> bool | None:
        """THE PRIMARY QUALITY READ. Did the engine get a single UNSEEN transition exactly right?

        This bar exists because the `cell_recall > 0.5` one below FAILED on this run's own data,
        and that failure is the most useful thing the run produced. tu93's best CONTROL engine
        scores `heldout_cell_recall 0.537` -- clearing the 0.5 line -- with **0 of 19 held-out
        changing transitions correct and 82 invented cells**. `cell_recall` is per-CELL overlap,
        so an engine that smears a roughly-right-sized change over roughly-right places clears a
        50% overlap line while never once reproducing a transition. It is the same class of trap
        as `usable`, one level up, and it admitted junk exactly as `usable` did in Phase 2.

        The strict bar is not gameable that way: `n_changes_correct >= 1` requires the engine to
        predict at least one unseen changed grid EXACTLY. Where the held-out set has no changing
        rows (the ft09/lp85 click games) it falls back to the no-op channel -- zero hallucinated
        no-ops -- which for a game where 76-92% of clicks are inert is the target-discrimination
        half of the mechanic. Where nothing at all is held out (vc33) it returns None.
        """
        if not m or not m.get("usable"):
            return False
        if m.get("heldout_n", 0) == 0:
            return None
        if m.get("heldout_n_changing", 0) > 0:
            return bool(m.get("heldout_n_changes_correct", 0) >= 1)
        return m.get("heldout_n_noop_hallucinated") == 0

    def _quality_ok(m: dict | None) -> bool | None:
        """Is this engine's OUT-OF-SAMPLE behaviour good, and not merely `usable`?

        Three cases, because the six games do not offer the same held-out evidence and
        pretending they do would be the whole error this phase exists to fix.

        1. HELD-OUT SET HAS CHANGING ROWS (tu93, tn36, sc25): grade on `heldout_cell_recall`
           against the reporting bar. This is the strong case -- it asks whether the engine
           predicts unseen CHANGES correctly.
        2. HELD-OUT SET IS ALL NO-OPS (ft09, lp85): these are click games where 76% resp. 92% of
           observed clicks are inert, so the mechanic IS target discrimination and the held-out
           question is whether the engine fires its rule at clicks the game ignores. Pass requires
           ZERO hallucinated no-ops -- and, separately, that the engine is not the identity
           function, which `usable` already established on the shown rows. Weaker than case 1 and
           labelled as such wherever it is reported.
        3. HELD-OUT SET IS EMPTY (vc33, whose real induce prompt carries ONE transition): None.
           Not a pass, not a fail. Nothing was measured.
        """
        if not m or not m.get("usable"):
            return False
        if m.get("heldout_n", 0) == 0:
            return None
        if m.get("heldout_cell_recall") is not None:
            return bool(m["heldout_cell_recall"] > QUALITY_RECALL_BAR)
        return m.get("heldout_n_noop_hallucinated") == 0

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
            # Rank by held-out cell_recall where that is measurable. Where it is NOT (a held-out
            # set of pure no-ops), fall back to held-out accuracy -- which on such a set is
            # exactly "did the engine leave the unseen no-ops alone". That is a real
            # out-of-sample signal about over-firing, and reporting nothing at all for those
            # games would throw it away; it is NOT evidence the engine models anything, because
            # the identity function scores 1.0 on it. Read it next to `engine_changes_anything`.
            best = None

            def _rank(m: dict):
                r = m.get("heldout_cell_recall")
                return r if r is not None else m.get("heldout_accuracy")

            for m in usable:
                if m.get("heldout_n", 0) == 0 or _rank(m) is None:
                    continue
                if best is None or _rank(m) > _rank(best):
                    best = m
            g[f"{arm}_best"] = (
                {
                    k: best.get(k)
                    for k in (
                        "tag",
                        "heldout_n",
                        "heldout_accuracy",
                        "heldout_cell_recall",
                        "heldout_change_fidelity",
                        "heldout_n_changing",
                        "heldout_n_changes_correct",
                        "heldout_invented_changed_cells",
                        "heldout_n_noop",
                        "heldout_n_noop_hallucinated",
                        "in_sample_cell_recall",
                        "in_sample_n_changes_correct",
                        "in_sample_n_changing",
                    )
                }
                if best
                else None
            )
            for suffix, fn_ in (("_any_quality", _quality_ok), ("_any_quality_strict", _quality_strict)):
                q = [fn_(m) for m in ms]
                g[f"{arm}{suffix}"] = (
                    True
                    if any(x is True for x in q)
                    else (None if any(x is None for x in q) else False)
                )
            g[f"{arm}_n_quality_strict"] = sum(1 for m in ms if _quality_strict(m) is True)
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
        return _quality_ok(m)

    tests = {
        "usable__treatment_final_vs_control": paired("TREATMENT_final", "CONTROL", _is_usable),
        "usable__treatment_round0_vs_control": paired("TREATMENT_round0", "CONTROL", _is_usable),
        "usable__treatment_final_vs_round0": paired(
            "TREATMENT_final", "TREATMENT_round0", _is_usable
        ),
        "quality_recall_bar__treatment_final_vs_control": paired(
            "TREATMENT_final", "CONTROL", _is_quality
        ),
        "quality_strict__treatment_final_vs_control": paired(
            "TREATMENT_final", "CONTROL", _quality_strict
        ),
    }

    n_games = len(games)
    funnel = {
        arm: {
            "games_any_usable": sum(1 for g in per_game if g[f"{arm}_any_usable"]),
            "games_any_quality": sum(1 for g in per_game if g[f"{arm}_any_quality"] is True),
            "games_any_quality_strict": sum(
                1 for g in per_game if g[f"{arm}_any_quality_strict"] is True
            ),
            "attempts_quality_strict": sum(g[f"{arm}_n_quality_strict"] for g in per_game),
            "games_quality_unscoreable": sum(
                1 for g in per_game if g[f"{arm}_any_quality"] is None
            ),
            "n_games": n_games,
        }
        for arm in ("CONTROL", "TREATMENT_round0", "TREATMENT_final")
    }

    # ---- THE VERDICT, computed rather than asserted -----------------------------------
    #
    # CONFIRMED requires BOTH halves, because either alone has a failure mode this repo has
    # already been bitten by:
    #   (a) the funnel reaches >= 3 of 6 games at the QUALITY bar, not merely at `usable`.
    #       Phase 2's tu93 engine was usable with 0/25 changes correct and 144 invented cells,
    #       so a usable-only funnel is not evidence of anything.
    #   (b) the treatment is not WORSE than control on the attempt-matched pairs. A funnel that
    #       reaches 3 games while losing paired attempts would mean the games were reached by
    #       resampling luck, which is what the p = 1.000 repair-vs-control result looked like.
    # UNDERPOWERED is reported when the funnel misses but the design could not have detected a
    # real effect anyway -- too few discordant pairs to reach significance, or too many games
    # whose held-out set cannot grade quality at all.
    t_usable = tests["usable__treatment_final_vs_control"]
    n_disc = t_usable["n_discordant"]
    # The smallest two-sided p this many discordant pairs could ever produce, if every one had
    # gone the same way. Reported BEFORE the result is read, per the phase's own rule.
    min_reachable_p = round(_sign_test(n_disc, 0), 4) if n_disc else 1.0
    q_final = funnel["TREATMENT_final"]
    q_ctl = funnel["CONTROL"]
    # STRICT bar is the one the verdict runs on. The recall bar is kept in the payload because
    # its failure on tu93 is itself a result, not because it is trusted.
    funnel_holds = q_final["games_any_quality_strict"] >= 3
    not_worse = t_usable["TREATMENT_final_only"] >= t_usable["CONTROL_only"]
    if funnel_holds and not_worse:
        verdict = "CONFIRMED"
    elif q_final["games_quality_unscoreable"] >= 3 or (n_disc and min_reachable_p > 0.05):
        verdict = "UNDERPOWERED"
    else:
        verdict = "NOT_CONFIRMED"

    # shipped accept-first behaviour, restated on this run's data
    ctl_rows = [a["CONTROL"] for a in attempts if a["CONTROL"]]
    accept_defective = sum(1 for m in ctl_rows if m["defect_kinds"])

    # COST. Reported because it can invert the conclusion's practical reading: a treatment that
    # needs a second call is not automatically more expensive if repetition control makes each
    # call stop naturally instead of grinding to the token cap. Counted per ATTEMPT (the unit an
    # operator pays for), so the treatment's re-ask is charged to the treatment.
    cost: dict = {}
    for label in ("CONTROL", "TREATMENT_round0", "TREATMENT_final"):
        # TREATMENT_final is charged for BOTH calls when the re-ask fired -- that is what an
        # operator pays. The re-ask row is recovered as the final row whenever it differs from
        # round 0, since the attempt record carries the arms by role rather than by call.
        tot_w = 0.0
        tot_t = tot_n = tot_c = 0
        for rec in attempts:
            if label == "TREATMENT_final":
                ms = [rec.get("TREATMENT_round0")]
                if rec.get("reask_fired") and rec.get("TREATMENT_final") is not rec.get(
                    "TREATMENT_round0"
                ):
                    ms.append(rec.get("TREATMENT_final"))
            else:
                ms = [rec.get(label)]
            for m in ms:
                if m and m.get("wall_s") is not None:
                    tot_w += float(m["wall_s"])
                    tot_t += int(m.get("predicted_n") or 0)
                    tot_c += 1 if m.get("stop_type") == "limit" else 0
                    tot_n += 1
        cost[label] = {
            "n_calls": tot_n,
            "total_wall_s": round(tot_w, 1),
            "wall_s_per_attempt": round(tot_w / max(1, len(attempts)), 1),
            "total_predicted_tokens": tot_t,
            "calls_hitting_token_cap": tot_c,
        }
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
        "verdict": verdict,
        "verdict_components": {
            "funnel_quality_strict_games_TREATMENT": q_final["games_any_quality_strict"],
            "funnel_quality_strict_games_CONTROL": q_ctl["games_any_quality_strict"],
            "funnel_quality_required": 3,
            "funnel_quality_recall_bar_games_TREATMENT": q_final["games_any_quality"],
            "funnel_quality_recall_bar_games_CONTROL": q_ctl["games_any_quality"],
            "games_total": q_final["n_games"],
            "games_quality_unscoreable_out_of_sample": q_final["games_quality_unscoreable"],
            "treatment_not_worse_on_paired_attempts": not_worse,
            "n_discordant_pairs_usable": n_disc,
            "min_reachable_p_two_sided": min_reachable_p,
            "observed_p_two_sided": t_usable["sign_test_p_two_sided"],
            "what_is_confirmed": (
                "The treatment reaches the strict out-of-sample quality bar on >=3 games and is "
                "never worse than control on the paired attempts. The measured effect is on "
                "engine VALIDITY and COST; see paired_tests for the quality channel, which is "
                "null at this n."
            ),
        },
        "n_attempts_total": len(attempts),
        "cost_per_arm": cost,
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
        f"{'game':5s} {'heldout':>16s}  {'arm':16s} {'usable':>7s} {'acc':>6s} {'recall':>7s} "
        f"{'fidel':>6s} {'chg ok':>9s} {'invent':>7s} {'noop_h':>8s}"
    )
    print(hdr)
    for g in per_game:
        ho = f"n={g['n_heldout']} chg={g['heldout_n_changing']}"
        for arm in ("CONTROL", "TREATMENT_round0", "TREATMENT_final"):
            b = g[f"{arm}_best"]
            n_us = g[f"{arm}_n_usable"]
            head = f"{g['game']:5s} {ho:>16s}  {arm:16s} {n_us:>3d}/{g['n_attempts']:<3d} "
            if b is None:
                print(head + f"{'--':>6s} {'--':>7s} {'--':>6s} {'--':>9s} {'--':>7s} {'--':>8s}")
                continue
            rec_s = (
                f"{b['heldout_cell_recall']:>7.3f}" if b["heldout_cell_recall"] is not None else f"{'n/a':>7s}"
            )
            fid_s = (
                f"{b['heldout_change_fidelity']:>6.3f}"
                if b["heldout_change_fidelity"] is not None
                else f"{'n/a':>6s}"
            )
            chg_s = f"{b['heldout_n_changes_correct']:>3d}/{b['heldout_n_changing']:<5d}"
            print(
                head + f"{b['heldout_accuracy']:>6.3f} {rec_s} {fid_s} {chg_s} "
                f"{b['heldout_invented_changed_cells']:>7d} "
                f"{b['heldout_n_noop_hallucinated']:>3d}/{b['heldout_n_noop']:<4d}"
            )
    print("\nfunnel (games with at least one attempt clearing the bar):")
    for arm, f in funnel.items():
        print(
            f"  {arm:16s} usable {f['games_any_usable']}/{f['n_games']}   "
            f"STRICT quality {f['games_any_quality_strict']}/{f['n_games']} "
            f"({f['attempts_quality_strict']}/{len(attempts) // 1} attempts)   "
            f"recall>{QUALITY_RECALL_BAR} bar {f['games_any_quality']}/{f['n_games']}   "
            f"({f['games_quality_unscoreable']} unscoreable)"
        )
    print("\ncost per attempt (the unit an operator pays; the re-ask is charged to TREATMENT):")
    for label, c in cost.items():
        print(
            f"  {label:16s} {c['n_calls']:>3d} calls  {c['wall_s_per_attempt']:>6.1f}s/attempt  "
            f"{c['total_predicted_tokens']:>7d} tok  "
            f"{c['calls_hitting_token_cap']}/{c['n_calls']} hit the token cap"
        )
    print("\npaired tests (attempt-matched):")
    for name, t in tests.items():
        print(
            f"  {name:42s} {t['arm_a']}_only={t[t['arm_a'] + '_only']} "
            f"{t['arm_b']}_only={t[t['arm_b'] + '_only']} both={t['both']} "
            f"neither={t['neither']}  p={t['sign_test_p_two_sided']}"
        )
    print(
        f"\nminimum reachable two-sided p at {n_disc} discordant pairs: {min_reachable_p} "
        f"(observed {t_usable['sign_test_p_two_sided']})"
    )
    print(
        f"\nVERDICT: {verdict}   "
        f"[STRICT quality funnel: treatment {q_final['games_any_quality_strict']}/"
        f"{q_final['n_games']} vs control {q_ctl['games_any_quality_strict']}/{q_ctl['n_games']}, "
        f"need 3; {q_final['games_quality_unscoreable']} unscoreable; "
        f"treatment_not_worse={not_worse}]"
    )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
