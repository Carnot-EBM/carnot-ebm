#!/usr/bin/env python3
"""PHASE 2, STEP 5 -- aggregate the per-game repair A/B into one artifact, and state the funnel.

THE ONLY NUMBER THAT JUDGES THIS WORK is how many of the five audited games produce a usable
engine that reaches the semantic gate. This script computes it from the per-game `ab.json` files
and refuses to compute anything for a game whose cell did not run: a game that was never launched
is a MISSING OBSERVATION and is reported as one, never as a zero. That distinction is the whole
reason the harness writes a per-game file rather than accumulating in memory.

`usable` is defined in `repair_ab.py` and is deliberately stricter than "returns on all paths":
mechanically clean AND the engine changes the grid on some observed transition. Phase 1's
best-scoring completions were the identity function, which clears the weaker bar trivially.
"""

from __future__ import annotations

import json
import pathlib
import time

REPO = pathlib.Path("/home/ianblenke/github.com/ianblenke/carnot")
HERE = pathlib.Path(__file__).resolve().parent
AB = HERE / "ab"
OUT = HERE.parent / "repair_ab.json"

GAMES = ["ft09", "tu93", "lp85", "tn36", "sc25"]


def main() -> int:
    per_game = {}
    for game in GAMES:
        p = AB / game / "ab.json"
        if not p.exists():
            per_game[game] = {"status": "NOT_RUN", "note": "cell never launched -- missing observation, not a zero"}
            continue
        d = json.loads(p.read_text())
        if d.get("status") != "ok":
            per_game[game] = {"status": d.get("status"), "witness": d.get("witness")}
            continue
        s = d["summary"]
        per_game[game] = {
            "status": "ok",
            "n_attempts": s["n_attempts"],
            "A_shipped_accepted": s["A_shipped_accepted"],
            "A_shipped_defective": s["A_shipped_defective"],
            "A_shipped_usable": s["A_shipped_usable"],
            "B_repair_usable": s["B_repair_usable"],
            "C_control_usable": s["C_control_usable"],
            "followups": s["followups"],
            "round0_defect_kinds": sorted(
                {k for a in d["attempts"] for k in (a.get("round0_defects") or [])}
            ),
            "n_transitions": d.get("n_transitions"),
            "prompt_sha256_16": d.get("prompt_sha256_16"),
            "server_exe_is_cuda_build": (d.get("witness") or {}).get("server_exe_is_cuda_build"),
            "vram_rows_mine": (d.get("witness") or {}).get("vram_rows_mine"),
            "wall_s": d.get("wall_s"),
        }

    ran = {g: v for g, v in per_game.items() if v.get("status") == "ok"}
    not_ran = [g for g, v in per_game.items() if v.get("status") != "ok"]

    def total(key: str) -> int:
        return sum(v[key] for v in ran.values())

    # "Repair produced 0 usable engines" and "there was nothing to repair" are DIFFERENT facts
    # and must never be added together. An attempt whose round 0 was already clean has no B or C
    # arm at all, so repair/control rates are computed only over attempts that HAD a defect.
    n_repairable_attempts = sum(
        1
        for v in ran.values()
        for f in v["followups"]
        if f == "repair_vs_control"
    )
    n_round0_clean = sum(
        1
        for v in ran.values()
        for f in v["followups"]
        if f == "skipped_round0_had_no_defects"
    )
    n_retry_attempts = sum(
        1 for v in ran.values() for f in v["followups"] if f == "retry_more_budget"
    )

    # A game "reaches the gate" in an arm if ANY attempt in that arm produced a usable engine.
    def games_with_usable(key: str) -> list[str]:
        return sorted(g for g, v in ran.items() if v[key] > 0)

    funnel = {
        "games_run": sorted(ran),
        "games_NOT_run": not_ran,
        "A_shipped_games_with_a_usable_engine": games_with_usable("A_shipped_usable"),
        "B_repair_games_with_a_usable_engine": games_with_usable("B_repair_usable"),
        "C_control_games_with_a_usable_engine": games_with_usable("C_control_usable"),
        "A_shipped_usable_attempts": total("A_shipped_usable"),
        "B_repair_usable_attempts": total("B_repair_usable"),
        "C_control_usable_attempts": total("C_control_usable"),
        "A_shipped_defective_attempts": total("A_shipped_defective"),
        "total_attempts": total("n_attempts"),
        # The denominators. Without these, "repair produced N usable engines" is unreadable.
        "attempts_with_a_repair_vs_control_pair": n_repairable_attempts,
        "attempts_whose_round0_was_already_clean": n_round0_clean,
        "attempts_that_took_the_truncation_retry_path": n_retry_attempts,
    }
    # The control is what separates "the repair TEXT works" from "any second ask works".
    b, c = funnel["B_repair_usable_attempts"], funnel["C_control_usable_attempts"]

    # THE COMPARISON IS PAIRED, AND COUNTING RAW WINS WOULD OVER-READ IT.
    # B and C branch from the SAME round 0, so each attempt is a matched pair and only the
    # DISCORDANT pairs (one arm usable, the other not) carry information about which arm is
    # better. An earlier version of this function declared "the DEFECT TEXT carries the value"
    # off a raw 3-vs-2 -- which an exact two-sided sign test puts at p = 1.000, i.e. exactly
    # what a coin does. That is the over-claim this project's own disciplines exist to stop,
    # and it was in the measurement tooling rather than in a result, which is worse.
    pairs = [
        (
            bool(a["B_repair"]["usable"]),
            bool(a["C_control"]["usable"]),
        )
        for g, v in ran.items()
        for a in json.loads((AB / g / "ab.json").read_text())["attempts"]
        if isinstance(a.get("B_repair"), dict) and isinstance(a.get("C_control"), dict)
    ]
    b_only = sum(1 for x, y in pairs if x and not y)
    c_only = sum(1 for x, y in pairs if y and not x)
    n_disc = b_only + c_only
    sign_p = 1.0
    if n_disc:
        import math

        sign_p = min(
            1.0,
            2
            * sum(math.comb(n_disc, k) for k in range(0, min(b_only, c_only) + 1))
            / 2**n_disc,
        )
    funnel["paired_discordant_pairs"] = n_disc
    funnel["paired_repair_only_wins"] = b_only
    funnel["paired_control_only_wins"] = c_only
    funnel["paired_sign_test_two_sided_p"] = round(sign_p, 4)

    if n_repairable_attempts == 0:
        verdict = (
            "NO attempt produced a repairable defect, so the repair arm was never exercised. "
            "This is a missing observation about repair, not evidence against it."
        )
    elif b == 0 and c == 0:
        verdict = (
            "NEITHER a defect-naming repair NOR a neutral re-ask produced a usable engine. "
            "The checks diagnose; they do not fix. The bottleneck is upstream of the feedback."
        )
    elif sign_p > 0.05:
        verdict = (
            f"A SECOND ASK RESCUES SOME ATTEMPTS; WHICH KIND OF ASK IS UNDECIDED. Repair "
            f"{b} usable vs control {c} over the same round-0 candidates, {n_disc} discordant "
            f"pairs ({b_only} repair-only, {c_only} control-only), exact two-sided sign test "
            f"p = {sign_p:.3f}. That is indistinguishable from a coin, so the DEFECT TEXT is "
            f"NOT shown to add anything over a contentless re-ask at this n. The finding that "
            f"survives is about arm A: the shipped path ACCEPTED "
            f"{funnel['A_shipped_defective_attempts']} defective candidates of "
            f"{funnel['total_attempts']} and therefore never re-asks at all."
        )
    elif b_only > c_only:
        verdict = (
            f"repair beat the control on paired discordant attempts ({b_only} vs {c_only}, "
            f"sign-test p = {sign_p:.3f}) -- the defect TEXT, not merely the second ask."
        )
    else:
        verdict = (
            f"the CONTROL beat the repair on paired discordant attempts ({c_only} vs "
            f"{b_only}, sign-test p = {sign_p:.3f}) -- do not attribute a win to the repair "
            f"content; echoing the failed code back may be actively harmful."
        )

    # ---- QUALITY, not just usability ------------------------------------------------------
    # `usable` (mechanically clean + not inert) is NECESSARY and nowhere near SUFFICIENT, and
    # the gate scores prove it on this very run: tu93's one "usable" repair has cell_recall
    # 0.112, change_fidelity 0.076, 144 INVENTED cells and 0 of 25 real changes right. It clears
    # the non-inert clause by changing the grid wrongly. Reporting the usable count alone would
    # therefore overstate what was achieved, so the production verifier's own cell_recall is
    # joined in here and a stricter tier is counted beside it.
    quality = {"note": "gate_scores.json not present -- run score_with_gate.py"}
    gs_path = HERE.parent / "gate_scores.json"
    if gs_path.exists():
        gs = json.loads(gs_path.read_text())
        scored = [r for r in gs["rows"] if r.get("status") == "ok"]
        strong = [r for r in scored if float(r.get("in_sample_cell_recall") or 0.0) > 0.5]
        quality = {
            "IN_SAMPLE_WARNING": gs["IN_SAMPLE_WARNING"],
            "bar": "in_sample_cell_recall > 0.5 -- models a real share of what reality changed",
            "n_completions_scored": len(scored),
            "strong_by_arm": {
                arm: sorted(r["game"] for r in strong if r["arm"] == arm)
                for arm in ("round0", "repair", "control")
            },
            "usable_but_degenerate": [
                {
                    "game": r["game"],
                    "arm": r["arm"],
                    "in_sample_cell_recall": r["in_sample_cell_recall"],
                    "invented_changed_cells": r["invented_changed_cells"],
                    "n_changes_correct": r["n_changes_correct"],
                    "n_changing": r["n_changing"],
                }
                for r in scored
                if r.get("usable") and float(r.get("in_sample_cell_recall") or 0.0) <= 0.5
            ],
            "best_completion": max(
                scored, key=lambda r: float(r.get("in_sample_cell_recall") or 0.0)
            ),
        }

    out = {
        "generated_by": "results/arc_engine_validation_20260731/harness/build_ab_artifact.py",
        "quality": quality,
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": (
            "Does feeding a MEASURED mechanical defect back to the generator produce a usable "
            "engine, and is any gain attributable to the defect TEXT rather than to the retry?"
        ),
        "usable_definition": (
            "generate() would accept it AND it carries no mechanical defect AND it changes the "
            "grid on some observed transition. The last clause is load-bearing: an identity "
            "engine clears the first two trivially (Phase 1)."
        ),
        "funnel": funnel,
        "verdict": verdict,
        "per_game": per_game,
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(funnel, indent=2, sort_keys=True))
    print("\nVERDICT:", verdict)
    if not_ran:
        print(f"\nNOT RUN (missing observations, not zeros): {not_ran}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
