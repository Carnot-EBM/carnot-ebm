#!/usr/bin/env python3
"""The pre-registered analysis: stratified permutation test, clustered at the GAME.

CLUSTERING IS AT THE GAME, NOT THE CELL, and that is not a stylistic choice. Replicates within
one game share a prompt, a window and a set of dynamics; treating them as independent trials
inflated a sibling experiment's p from 0.125 to 0.049 on 2026-07-31 and had to be corrected.
Here the arm label is permuted WITHIN each game, which respects the clustering exactly: the
null is "the arm label is exchangeable inside a game", not "all cells are exchangeable".

THE ARMEDNESS GATE RUNS FIRST AND CAN VOID THE WHOLE RUN. If `goal_defect_reasks_delta` is 0
across the entire treatment arm, the treatment never fired, and the correct report is
NON-TEST, not null. exp6013 shipped a HUD-mask factor that was a silent no-op on all 162 arms
and was read as "both mask settings measured"; that is the failure this refuses to repeat.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
N_PERM = 200_000
# PRIMARY SWAPPED BEFORE ANY LLM CALL, see the pre-registration's AMENDMENT_2. The former
# primary `O4_discriminates_heldout` is DETERMINED by the treatment's own accept decision --
# every predicate the gate keeps scores O4-positive (6 of 6, FN=0, over 115 frozen engines) --
# which is the same circularity as scoring against plan_found. It is demoted, not deleted, and
# is still reported in full so the swap is auditable.
PRIMARY = "O6_pre_win_and_not_open"
SECONDARIES = [
    "O4_discriminates_heldout",
    "O2_fires_pre_win",
    "O1_fires_post_win",
    "O3_fires_heldout",
    "O7b_all_false_observed",
]


def game_rates(cells: list[dict], key: str, arm_tag: str) -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    for c in cells:
        if c["tag"] == arm_tag and c.get("outcomes"):
            by[c["game"]].append(float(bool(c["outcomes"][key])))
    return {g: float(np.mean(v)) for g, v in by.items() if v}


def perm_test(cells: list[dict], key: str, tag_a: str, tag_b: str, seed: int = 12345) -> dict:
    """Statistic = mean over games of (rate_b - rate_a). Arm label permuted WITHIN game."""
    rng = np.random.default_rng(seed)
    per_game: dict[str, tuple[list[float], list[float]]] = {}
    for c in cells:
        if not c.get("outcomes"):
            continue
        if c["tag"] not in (tag_a, tag_b):
            continue
        a, b = per_game.setdefault(c["game"], ([], []))
        (a if c["tag"] == tag_a else b).append(float(bool(c["outcomes"][key])))
    games = [g for g, (a, b) in per_game.items() if a and b]
    if not games:
        return {"n_games": 0, "p": None, "reason": "no game has both arms measurable"}
    obs = float(np.mean([np.mean(per_game[g][1]) - np.mean(per_game[g][0]) for g in games]))
    pooled = {g: np.array(per_game[g][0] + per_game[g][1]) for g in games}
    na = {g: len(per_game[g][0]) for g in games}
    cnt = 0
    for _ in range(N_PERM):
        diffs = []
        for g in games:
            v = rng.permutation(pooled[g])
            diffs.append(v[na[g] :].mean() - v[: na[g]].mean())
        if abs(float(np.mean(diffs))) >= abs(obs) - 1e-12:
            cnt += 1
    return {
        "n_games": len(games),
        "rate_a": round(float(np.mean([np.mean(per_game[g][0]) for g in games])), 4),
        "rate_b": round(float(np.mean([np.mean(per_game[g][1]) for g in games])), 4),
        "observed_effect": round(obs, 4),
        "p": round((cnt + 1) / (N_PERM + 1), 5),
        "n_discordant_games": sum(
            1 for g in games if abs(np.mean(per_game[g][1]) - np.mean(per_game[g][0])) > 1e-12
        ),
        "games_favouring_b": sum(
            1 for g in games if np.mean(per_game[g][1]) > np.mean(per_game[g][0])
        ),
        "games_favouring_a": sum(
            1 for g in games if np.mean(per_game[g][1]) < np.mean(per_game[g][0])
        ),
    }


def main() -> int:
    cells = json.loads((HERE / "out" / "scored.json").read_text())
    out: dict = {}

    # ---- missingness, reported per arm, never as zeros ----
    miss: dict[str, dict] = {}
    for tag in ("off", "on", "aa"):
        arm = [c for c in cells if c["tag"] == tag]
        miss[tag] = {
            "n_cells": len(arm),
            "n_measurable": sum(1 for c in arm if c.get("outcomes")),
            "n_induce_failed": sum(1 for c in arm if not c.get("induce_ok")),
            "n_server_failure": sum(1 for c in arm if c.get("server_failures_delta", 0) > 0),
            "n_content_failure": sum(1 for c in arm if c.get("content_failures_delta", 0) > 0),
            "n_goal_unscorable": sum(
                1 for c in arm if not c.get("outcomes") and c.get("induce_ok")
            ),
            "goal_scorer_status": {
                s: sum(1 for c in arm if c.get("goal_raw", {}).get("status") == s)
                for s in {c.get("goal_raw", {}).get("status") for c in arm}
            },
        }
    out["missingness"] = miss
    out["missing_is_never_zero"] = (
        "a cell whose induce raised, whose server failed, or whose goal scorer timed out is "
        "EXCLUDED and counted above. It is never scored 0."
    )

    # ---- ARMEDNESS GATE ----
    on_reasks = sum(c.get("goal_defect_reasks_delta", 0) for c in cells if c["tag"] == "on")
    off_reasks = sum(c.get("goal_defect_reasks_delta", 0) for c in cells if c["tag"] != "on")
    out["armedness"] = {
        "goal_defect_reasks_in_treatment": int(on_reasks),
        "goal_defect_reasks_in_control_arms": int(off_reasks),
        "treatment_fired": on_reasks > 0,
        "control_stayed_inert": off_reasks == 0,
        "cells_where_gate_fired": sum(
            1 for c in cells if c["tag"] == "on" and c.get("goal_defect_reasks_delta", 0) > 0
        ),
        "verdict": (
            "ARMED"
            if on_reasks > 0 and off_reasks == 0
            else ("NON_TEST_treatment_never_fired" if on_reasks == 0 else "CONTAMINATED_control")
        ),
    }

    # ---- PRIMARY + A/A, same test ----
    out["PRIMARY"] = {
        "metric": PRIMARY,
        "on_vs_off": perm_test(cells, PRIMARY, "off", "on"),
        "AA_noise_floor_aa_vs_off": perm_test(cells, PRIMARY, "off", "aa"),
    }
    out["SECONDARIES"] = {
        k: {"on_vs_off": perm_test(cells, k, "off", "on"), "AA": perm_test(cells, k, "off", "aa")}
        for k in SECONDARIES
    }

    # ---- GUARDRAIL: did the engine degrade? ----
    def eng_rates(tag: str, field: str) -> dict[str, float]:
        by: dict[str, list[float]] = defaultdict(list)
        for c in cells:
            e = c.get("engine") or {}
            if c["tag"] == tag and e.get("measurable"):
                by[c["game"]].append(float(e.get(field, 0.0)))
        return {g: float(np.mean(v)) for g, v in by.items() if v}

    guard = {}
    for field in ("change_fidelity", "cell_recall", "accuracy"):
        a, b = eng_rates("off", field), eng_rates("on", field)
        shared = sorted(set(a) & set(b))
        guard[field] = {
            "n_games": len(shared),
            "off": round(float(np.mean([a[g] for g in shared])), 4) if shared else None,
            "on": round(float(np.mean([b[g] for g in shared])), 4) if shared else None,
            "delta": round(float(np.mean([b[g] - a[g] for g in shared])), 4) if shared else None,
            "games_engine_worse_under_treatment": sum(1 for g in shared if b[g] < a[g] - 1e-9),
            "games_engine_better_under_treatment": sum(1 for g in shared if b[g] > a[g] + 1e-9),
        }
    out["GUARDRAIL_engine"] = guard

    # ---- determinism witness: paired arms share attempt 0, so identical seeds should agree ----
    by_key = {(c["game"], c["replicate"], c["tag"]): c for c in cells}
    same, diff = 0, 0
    for (g, r, tag), c in by_key.items():
        if tag != "on":
            continue
        ctl = by_key.get((g, r, "off"))
        if ctl and c.get("engine_sha256") and ctl.get("engine_sha256"):
            if c["engine_sha256"] == ctl["engine_sha256"]:
                same += 1
            else:
                diff += 1
    out["pairing_witness"] = {
        "on_identical_to_off": same,
        "on_differs_from_off": diff,
        "reading": "control and treatment send the IDENTICAL seed and combined prompt, so a "
        "pair is identical exactly where the gate did not fire. `on_differs_from_off` "
        "should closely track cells_where_gate_fired; a large excess would mean the "
        "sampler is not actually deterministic and the pairing claim is weaker than stated.",
    }

    (HERE / "out" / "analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
