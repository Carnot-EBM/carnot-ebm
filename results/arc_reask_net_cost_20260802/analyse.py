#!/usr/bin/env python3
"""Score every cell for USABILITY (killable subprocess) and run the pre-registered analysis.

THE PRIMARY IS `net = usable - hard_failures`, clustered at the GAME level. Both components are
also reported separately, because the whole point of this run is that the gate's historical
headline reported only the first one.

CLUSTERING IS NOT OPTIONAL. Replicates of one game are replicates of ONE PROMPT, not independent
trials. Treating them as independent inflated a p from 0.125 to 0.049 in a sibling run on
2026-08-01 and had to be corrected. Everything below aggregates within a game first.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = os.environ.get("RNC_PY", "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python")
SCRATCH = pathlib.Path(
    "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/reaskcost"
)
WINDOW_DIR = SCRATCH.parent / "goalab" / "windows"
OUT = HERE / os.environ.get("RNC_OUT_SUBDIR", "out")
SCORE_TIMEOUT = 300
TAGS = ["a_off", "b_shipped", "c_owns", "aa"]


def score_one(row: dict) -> dict:
    """Score ONE cell. Returns the row with `usable` / `hard_failure` / `measurable` attached."""
    r = dict(row)
    # ---- missingness FIRST. A missing observation is never a zero and never a -1. ----
    if r.get("exception"):
        r.update(measurable=False, miss_reason="driver_exception")
        return r
    if r.get("server_failures_delta", 0) > 0:
        r.update(measurable=False, miss_reason="server_failure")
        return r
    if not r.get("induce_ok"):
        # A CONTENT failure: the server answered every try and the answers were unusable. This
        # is the cost column the gate's historical headline omitted. It is a REAL -1.
        r.update(measurable=True, usable=False, hard_failure=True, net=-1, defect_kinds=None)
        return r
    eng = pathlib.Path(r["e3_dir"]) / r["game"] / "world_model.py"
    if not eng.exists():
        # induce reported success but wrote nothing -- treat as unmeasurable rather than
        # inventing a verdict about a file that is not there.
        r.update(measurable=False, miss_reason="induce_ok_but_no_engine_file")
        return r
    job = SCRATCH / f"score_{r['game']}_{r['replicate']}_{r['tag']}.json"
    job.parent.mkdir(parents=True, exist_ok=True)
    job.write_text(
        json.dumps(
            {
                "engine_path": str(eng),
                "window_pkl": str(WINDOW_DIR / f"{r['game']}.pkl"),
                "budget": r.get("max_tokens", 4096),
            }
        )
    )
    env = {
        "CARNOT_REPO": str(REPO),
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ.get("HOME", "/home/ianblenke"),
        "CARNOT_ARC_OFFLINE": "1",
        "CARNOT_ARC_E3_DIR": str(SCRATCH / "score_e3"),
    }
    try:
        p = subprocess.run(
            [PY, str(HERE / "usable_worker.py"), str(job)],
            capture_output=True,
            text=True,
            timeout=SCORE_TIMEOUT,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        r.update(measurable=False, miss_reason="usable_scorer_timeout")
        return r
    out = None
    for line in reversed((p.stdout or "").strip().splitlines()):
        try:
            out = json.loads(line)
            break
        except Exception:  # noqa: BLE001,S112
            continue
    if not out or not out.get("scored"):
        r.update(
            measurable=False,
            miss_reason="usable_scorer_failed",
            scorer_stderr=(p.stderr or "")[-300:],
        )
        return r
    usable = bool(out["usable"])
    r.update(
        measurable=True,
        usable=usable,
        hard_failure=False,
        net=1 if usable else 0,
        defect_kinds=out.get("defect_kinds"),
    )
    return r


def sign_test(pairs: list[tuple[float, float]]) -> dict:
    """Two-sided EXACT sign test on paired per-game values. `pairs` is [(x_game, y_game), ...]."""
    pos = sum(1 for x, y in pairs if y > x)  # favours y (the second arm)
    neg = sum(1 for x, y in pairs if y < x)
    n = pos + neg
    if n == 0:
        p = 1.0
    else:
        k = min(pos, neg)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
        p = min(1.0, 2.0 * tail)
    return {
        "n_games_paired": len(pairs),
        "n_discordant_games": n,
        "games_favouring_second": pos,
        "games_favouring_first": neg,
        "mean_first": round(sum(x for x, _ in pairs) / len(pairs), 4) if pairs else None,
        "mean_second": round(sum(y for _, y in pairs) / len(pairs), 4) if pairs else None,
        "observed_effect": round(
            (sum(y for _, y in pairs) - sum(x for x, _ in pairs)) / len(pairs), 4
        )
        if pairs
        else None,
        "p": round(p, 6),
        "min_reachable_p_at_this_n_discordant": round(min(1.0, 2.0 * 0.5**n), 9) if n else 1.0,
    }


def main() -> int:
    rows = json.loads((OUT / "rows.json").read_text())
    with ThreadPoolExecutor(max_workers=4) as ex:
        scored = list(ex.map(score_one, rows))
    (OUT / "scored.json").write_text(json.dumps(scored, indent=1))

    by_tag: dict[str, list[dict]] = {t: [] for t in TAGS}
    for r in scored:
        by_tag.setdefault(r["tag"], []).append(r)

    # ---- missingness + the two RAW components, per arm ----
    missingness, components = {}, {}
    for tag, rs in by_tag.items():
        meas = [r for r in rs if r.get("measurable")]
        miss: dict[str, int] = {}
        for r in rs:
            if not r.get("measurable"):
                miss[r.get("miss_reason", "unknown")] = (
                    miss.get(r.get("miss_reason", "unknown"), 0) + 1
                )
        usable = sum(1 for r in meas if r.get("usable"))
        hard = sum(1 for r in meas if r.get("hard_failure"))
        missingness[tag] = {
            "n_cells": len(rs),
            "n_measurable": len(meas),
            "n_missing": len(rs) - len(meas),
            "missing_by_reason": miss,
        }
        components[tag] = {
            "n_measurable": len(meas),
            "usable": usable,
            "succeeded_but_defective": len(meas) - usable - hard,
            "hard_failures": hard,
            "net": usable - hard,
            "usable_rate": round(usable / len(meas), 4) if meas else None,
            "hard_failure_rate": round(hard / len(meas), 4) if meas else None,
        }

    # ---- armedness: the gate must FIRE in b/c and never in a/aa ----
    fired = {t: sum(r.get("engine_defect_reasks_delta", 0) for r in rs) for t, rs in by_tag.items()}
    cells_fired = {
        t: sum(1 for r in rs if r.get("engine_defect_reasks_delta", 0) > 0)
        for t, rs in by_tag.items()
    }
    armedness = {
        "engine_defect_reasks_total_by_arm": fired,
        "cells_where_gate_fired_by_arm": cells_fired,
        "treatment_fired": fired.get("b_shipped", 0) > 0 and fired.get("c_owns", 0) > 0,
        "control_stayed_inert": fired.get("a_off", 0) == 0 and fired.get("aa", 0) == 0,
        "goal_gate_stayed_inert": all(r.get("goal_defect_reasks_delta", 0) == 0 for r in scored),
        "arm_flags_consistent_everywhere": all(r.get("arm_flags_consistent") for r in scored),
    }
    # TWO SEPARATE QUESTIONS, deliberately not collapsed into one verdict. Collapsing them is
    # how a real finding gets mislabelled a broken experiment.
    #   CONFIGURATION -- did the arms actually differ? Proven BEFORE the run by the treatment
    #     witness (per-arm re-ask budget 0/1/1, owns-attempts false/false/true, and the detector
    #     biting on a known-defective engine). If this fails, the run is a non-test.
    #   EXPOSURE -- did the gate ever FIND a defect to act on? This is a RESULT, not a
    #     configuration property. Zero exposure with a proven-different configuration means the
    #     gate is INERT on this stack -- which is an answer, not a failure.
    armedness["configuration_verdict"] = (
        "CONFIGURED"
        if armedness["control_stayed_inert"]
        and armedness["goal_gate_stayed_inert"]
        and armedness["arm_flags_consistent_everywhere"]
        else "MISCONFIGURED"
    )
    armedness["exposure_verdict"] = (
        "GATE_FIRED" if armedness["treatment_fired"] else "ZERO_EXPOSURE"
    )
    armedness["verdict"] = (
        "ARMED_AND_FIRED"
        if armedness["configuration_verdict"] == "CONFIGURED" and armedness["treatment_fired"]
        else (
            "CONFIGURED_BUT_ZERO_EXPOSURE"
            if armedness["configuration_verdict"] == "CONFIGURED"
            else "MISCONFIGURED_NON_TEST"
        )
    )
    armedness["reading"] = (
        "CONFIGURED_BUT_ZERO_EXPOSURE is a FINDING, not a broken run: the arms provably differed "
        "and the detector provably bites, so the gate had every opportunity to act and never "
        "found a defect to act on. What it does mean is that the per-firing COST of the gate is "
        "unobservable here -- you cannot measure the price of an event that did not occur -- so "
        "the primary contrast bounds the gate's TOTAL cost (at zero exposure, zero) without "
        "estimating its cost per firing."
    )

    # ---- INTERNAL VALIDITY: is the pairing real? ----
    # a_off, b_shipped and c_owns send the IDENTICAL prompt at the IDENTICAL seed. So in any
    # (game, replicate) where NEITHER arm's gate fired, the two arms should have produced the
    # BYTE-IDENTICAL engine. If they did not, the sampler is not reproducible at fixed seed and
    # the "paired" contrast is really two independent draws -- which would not invalidate the
    # sign test, but would mean the A/A floor is the only honest yardstick for effect size.
    # Reported either way rather than assumed.
    idx = {(r["game"], r["replicate"], r["tag"]): r for r in scored}
    pairing = {}
    # b_shipped vs c_owns is the DECISIVE diagnostic, not an afterthought. Those two arms differ
    # ONLY in whether a defect re-ask grants or consumes an attempt, so in a cell where NO defect
    # fired they are the same code path at the same seed at ADJACENT request positions. If they
    # come back byte-identical while a_off (request position 1, cold prompt cache) diverges, the
    # sampler IS deterministic at fixed seed and the divergence is REQUEST ORDER / prompt-cache
    # state -- not broken seeding.
    for base, other in (("a_off", "b_shipped"), ("a_off", "c_owns"), ("b_shipped", "c_owns")):
        same, diff, checked = 0, 0, []
        for (g, rep, tag), r in idx.items():
            if tag != base:
                continue
            o = idx.get((g, rep, other))
            if o is None:
                continue
            if r.get("engine_defect_reasks_delta", 0) or o.get("engine_defect_reasks_delta", 0):
                continue  # a gate fired: divergence is the treatment, not the sampler
            a_sha, o_sha = r.get("engine_sha256"), o.get("engine_sha256")
            if not a_sha or not o_sha:
                continue
            checked.append((g, rep))
            if a_sha == o_sha:
                same += 1
            else:
                diff += 1
        pairing[f"{base}_vs_{other}"] = {
            "n_cells_where_no_gate_fired": len(checked),
            "byte_identical_engine": same,
            "diverged": diff,
            "identical_rate": round(same / len(checked), 4) if checked else None,
        }
    pairing["reading"] = (
        "these arms send the identical prompt at the identical seed, so where no gate fired "
        "they SHOULD be byte-identical. Read b_shipped_vs_c_owns FIRST: those two are adjacent "
        "request positions on the same code path, so if THEY match while a_off (request "
        "position 1, cold prompt cache) diverges, the seed is working and the divergence is "
        "request-order / prompt-cache state. That degrades the nominal pairing to independent "
        "draws -- it does NOT bias either arm, because cache state changes which sample you get "
        "and not the distribution it is drawn from -- and it makes the A/A floor, rather than "
        "the nominal pairing, the operative yardstick for effect size. Which is exactly why an "
        "A/A arm is mandatory on this path."
    )

    # ---- per-game aggregation (THE clustering step) then the paired sign tests ----
    def game_totals(tag: str) -> dict[str, dict]:
        acc: dict[str, dict] = {}
        for r in by_tag.get(tag, []):
            if not r.get("measurable"):
                continue
            a = acc.setdefault(r["game"], {"net": 0, "usable": 0, "hard": 0, "n": 0})
            a["net"] += r["net"]
            a["usable"] += 1 if r.get("usable") else 0
            a["hard"] += 1 if r.get("hard_failure") else 0
            a["n"] += 1
        return acc

    totals = {t: game_totals(t) for t in TAGS}

    def contrast(first: str, second: str, key: str) -> dict:
        # A game enters the pairing only if BOTH arms have at least one measurable replicate of
        # it. A game measurable in one arm only cannot be paired without inventing the other
        # side, and inventing it as a zero is exactly the fail-as-zero error this design forbids.
        games = sorted(set(totals[first]) & set(totals[second]))
        pairs = [(totals[first][g][key], totals[second][g][key]) for g in games]
        out = sign_test(pairs)
        out["per_game"] = {
            g: {first: totals[first][g][key], second: totals[second][g][key]} for g in games
        }
        out["games_dropped_unpaired"] = sorted(
            (set(totals[first]) | set(totals[second])) - set(games)
        )
        return out

    # ---- MECHANISM / EXPOSURE. The marginal contrast is diluted by every cell where the gate
    # never fired: in those cells b_shipped IS a_off's code path, so they contribute noise and no
    # signal. The damage mechanism -- a re-ask spends the attempt that would have been the accept
    # -- can only operate in a cell where the gate FIRED. So the gate's exposure (how often it
    # fires) is as load-bearing as its per-firing cost, and it is what distinguishes this gate
    # from the goal gate that attrited 17 of 21 cells: that one fired on ~76% of cells because
    # ~89% of induced goals are constant, and this one can only fire on a mechanically defective
    # ENGINE.
    def exposure(tag: str) -> dict:
        rs = [r for r in by_tag.get(tag, []) if r.get("measurable") or r.get("exception") is None]
        fired_cells = [r for r in rs if r.get("engine_defect_reasks_delta", 0) > 0]
        meas_fired = [r for r in fired_cells if r.get("measurable")]
        return {
            "n_cells": len(rs),
            "n_cells_where_gate_fired": len(fired_cells),
            "exposure_rate": round(len(fired_cells) / len(rs), 4) if rs else None,
            "given_fired_hard_failures": sum(1 for r in meas_fired if r.get("hard_failure")),
            "given_fired_usable": sum(1 for r in meas_fired if r.get("usable")),
            "given_fired_n_measurable": len(meas_fired),
            "given_NOT_fired_hard_failures": sum(
                1
                for r in rs
                if r.get("measurable")
                and not r.get("engine_defect_reasks_delta", 0)
                and r.get("hard_failure")
            ),
        }

    mechanism = {t: exposure(t) for t in TAGS}
    # The PAIRED mechanism view: in exactly the cells where b_shipped's gate fired, what did the
    # gate-off arm do with the same prompt and seed? This is the only place the trade can be seen
    # directly rather than inferred from a diluted marginal.
    paired_fired = []
    for (g, rep, tag), r in sorted(idx.items()):
        if tag != "b_shipped" or not r.get("engine_defect_reasks_delta", 0):
            continue
        a = idx.get((g, rep, "a_off"))
        c = idx.get((g, rep, "c_owns"))
        paired_fired.append(
            {
                "game": g,
                "replicate": rep,
                "b_reasks": r.get("engine_defect_reasks_delta"),
                "b_outcome": "hard_failure"
                if r.get("hard_failure")
                else ("usable" if r.get("usable") else "defective"),
                "a_off_outcome": None
                if a is None or not a.get("measurable")
                else (
                    "hard_failure"
                    if a.get("hard_failure")
                    else ("usable" if a.get("usable") else "defective")
                ),
                "c_owns_outcome": None
                if c is None or not c.get("measurable")
                else (
                    "hard_failure"
                    if c.get("hard_failure")
                    else ("usable" if c.get("usable") else "defective")
                ),
                "c_reasks": None if c is None else c.get("engine_defect_reasks_delta"),
            }
        )

    analysis = {
        "MECHANISM_and_exposure": {
            "why": "the marginal primary is diluted by every cell where the gate never fired -- "
            "in those cells b_shipped IS a_off's code path. The damage mechanism (a re-ask "
            "spends the attempt that would have been the accept) can ONLY operate where the gate "
            "fired, so exposure is as load-bearing as per-firing cost.",
            "by_arm": mechanism,
            "paired_cells_where_b_shipped_gate_fired": paired_fired,
        },
        "PRIMARY": {
            "metric": "net = usable - hard_failures, summed within a game then paired across arms",
            "b_shipped_vs_a_off": contrast("a_off", "b_shipped", "net"),
            "c_owns_vs_a_off": contrast("a_off", "c_owns", "net"),
            "c_owns_vs_b_shipped": contrast("b_shipped", "c_owns", "net"),
            "AA_noise_floor_aa_vs_a_off": contrast("a_off", "aa", "net"),
        },
        "ROBUSTNESS_to_request_position": {
            "why": "cells run in the fixed order a_off, b_shipped, c_owns, aa, so a_off is "
            "ALWAYS request position 1 for a given game and pays a cold prompt cache while the "
            "others run warm. That changes which sample is drawn (see "
            "pairing_internal_validity) though not the distribution it comes from. `aa` is a "
            "gate-OFF arm at request position 4, so it BRACKETS b_shipped's position 2 from the "
            "other side: if b_vs_a and b_vs_aa point the same way, the primary is not an "
            "artefact of request order.",
            "b_shipped_vs_a_off_position_1_control": contrast("a_off", "b_shipped", "net"),
            "b_shipped_vs_aa_position_4_control": contrast("aa", "b_shipped", "net"),
            "c_owns_vs_aa_position_4_control": contrast("aa", "c_owns", "net"),
        },
        "SECONDARY_usable_only": {
            "note": "the gate's OWN historical metric. Reported so a reader can see which "
            "component moved -- NOT the primary, because scoring on usable alone is exactly "
            "what hid this defect.",
            "b_shipped_vs_a_off": contrast("a_off", "b_shipped", "usable"),
            "c_owns_vs_a_off": contrast("a_off", "c_owns", "usable"),
            "AA_noise_floor": contrast("a_off", "aa", "usable"),
        },
        "SECONDARY_hard_failures_only": {
            "note": "the component the historical metric omitted. Here MORE is WORSE, so a "
            "positive observed_effect favours the FIRST arm, not the second.",
            "b_shipped_vs_a_off": contrast("a_off", "b_shipped", "hard"),
            "c_owns_vs_a_off": contrast("a_off", "c_owns", "hard"),
            "AA_noise_floor": contrast("a_off", "aa", "hard"),
        },
        "components_by_arm": components,
        "missingness": missingness,
        "missing_is_never_zero": "a cell whose server failed, whose driver raised, or whose "
        "usability scorer timed out is EXCLUDED and counted above. It is never scored 0 or -1. "
        "A CONTENT failure is different: the server answered and the answers were unusable, "
        "which is a real -1 and is the cost this experiment exists to count.",
        "armedness": armedness,
        "pairing_internal_validity": pairing,
        "clustering": "GAME. Replicates within a game are replicates of ONE PROMPT. Treating "
        "them as independent trials inflated a sibling run's p from 0.125 to 0.049 on "
        "2026-08-01 and had to be corrected.",
    }
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=1))
    print(json.dumps({"components_by_arm": components, "armedness": armedness}, indent=1))
    for name, c in analysis["PRIMARY"].items():
        if isinstance(c, dict) and "p" in c:
            print(
                f"PRIMARY {name}: effect={c['observed_effect']} p={c['p']} "
                f"disc={c['n_discordant_games']} "
                f"(min reachable {c['min_reachable_p_at_this_n_discordant']})"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
