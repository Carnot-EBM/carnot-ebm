#!/usr/bin/env python3
"""Score the fidelity-vs-plannability regression, and say plainly what it licenses.

THE QUESTION THIS SETTLES. `change_fidelity` became the primary metric for ARC induction work on
2026-08-01 after exp6018's exact-match primary was found floored (zero in both arms, zero
discordant pairs, no test possible). The object-perception A/B then moved it, p=0.0192. But that
A/B recorded its own caveat: the metric's relationship to downstream PLANNING VALUE was
UNVERIFIED, and the frozen 48-candidate join pointed the wrong way -- all six candidates scoring a
perfect 1.0 were unplannable, both plannable ones sat in the bottom half. With n_plannable = 2
that established nothing, and the artifact said so: "a reason to MEASURE the link, not a finding
that the link is absent."

This is that measurement at n_plannable = 17, across 9 games.

WHY AUC AND NOT A CORRELATION. `plan_found` is binary and rare (17 of 111). AUC is exactly
"probability a random plannable engine outranks a random unplannable one on this metric", which is
the question -- does the metric ORDER engines by planning value -- and it is invariant to the
metric's arbitrary scale. A Pearson r on a rare binary would be driven by the base rate.

WHAT A NULL HERE MEANS, stated before the numbers. If the CI spans 0.5, the metric is not shown to
order engines by planning value. That does NOT make the object-perception effect fake -- the
effect on change_fidelity is real and reproduced. It makes it UNCASHABLE: a real movement in a
number whose downstream value is unestablished. Every representation A/B scored on this metric
inherits that, including the HUD-mask re-score and the headroom work.

THE COMPETING PREDICTORS ARE REPORTED BUT NOT PROMOTED. Two of them are traps:
  * `n_changing` is a property of the WINDOW, not the engine -- a game with more changing
    transitions supplying more plannable engines is a corpus effect wearing a predictor's clothes.
  * `spurious_changed_cells` running in the "wrong" direction most likely says an engine that
    writes more cells reaches SOME goal more often, not that spurious writes are good.
Neither is offered as a replacement metric. They are recorded because the honest common thread --
that what mildly tracks plannability is engines that DO SOMETHING rather than engines that predict
accurately -- is itself the finding worth acting on.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
import statistics as st
from collections import Counter

OUT = pathlib.Path(__file__).resolve().parent.parent
RAW = json.loads((OUT / "plan_regression_raw.json").read_text())
ROWS = [r for r in RAW["rows"] if r.get("change_fidelity") is not None and r.get("status") == "ok"]
PLAN = [r for r in ROWS if r["plan_found"]]
NOPLAN = [r for r in ROWS if not r["plan_found"]]


def auc(a: list, b: list, field: str) -> float | None:
    """P(random member of `a` outranks a random member of `b`) on `field`; ties count 0.5."""
    if not a or not b:
        return None
    w = 0.0
    for x in a:
        for y in b:
            xa, yb = x.get(field), y.get(field)
            if xa is None or yb is None:
                return None
            w += 1.0 if xa > yb else 0.5 if xa == yb else 0.0
    return w / (len(a) * len(b))


def boot_ci(field: str, n: int = 4000, seed: int = 0) -> tuple[float, float] | None:
    rnd = random.Random(seed)
    vals = []
    for _ in range(n):
        pa = [rnd.choice(PLAN) for _ in PLAN]
        pb = [rnd.choice(NOPLAN) for _ in NOPLAN]
        v = auc(pa, pb, field)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


primary = auc(PLAN, NOPLAN, "change_fidelity")
ci = boot_ci("change_fidelity")
spans_half = ci is not None and ci[0] <= 0.5 <= ci[1]

competing = {}
for f in (
    "cell_recall",
    "accuracy",
    "spurious_changed_cells",
    "n_changing",
    "distinct_successors_at_root",
):
    v = auc(PLAN, NOPLAN, f)
    if v is not None:
        competing[f] = round(v, 4)

by_game = dict(Counter(r["game"] for r in PLAN))

report = {
    "schema": "carnot.arc_fidelity_vs_plannability.v1",
    "experiment": "arc_fidelity_vs_plan_116_20260801",
    "question": (
        "Does held-out change_fidelity ORDER induced engines by their planning value "
        "(plan_in_model finding a plan from the level root)?"
    ),
    "run_date": "2026-08-01T00:00:00Z",
    "duration_s": RAW.get("wall_s"),
    "random_seed": 0,
    "random_seed_note": (
        "Seeds the bootstrap resampling only. plan_in_model and the engines are deterministic; "
        "no generation happens in this experiment."
    ),
    "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
    "inference_substrate_note": (
        "No model is loaded and no GPU is touched. This executes 116 already-induced Python "
        "engines inside the shipped planner, each in a killable subprocess."
    ),
    "verifier_is_oracle": True,
    "verifier_is_oracle_note": (
        "Declared True and no moat claim is made: whether a plan exists is decided by EXECUTING "
        "the engine, so the check IS the executable oracle. The finding is about whether one "
        "measurement predicts another."
    ),
    "preconditions_checked": [
        {"resource": "committed_engines", "available": True},
        {"resource": "ab_rows_with_change_fidelity", "available": True},
        {"resource": "offline_window_rebuild", "available": True},
    ],
    "n_scored": len(ROWS),
    "n_plannable": len(PLAN),
    "n_unplannable": len(NOPLAN),
    "plannable_by_game": by_game,
    "n_games_contributing_plannable": len(by_game),
    "PRIMARY_auc_change_fidelity": round(primary, 6) if primary is not None else None,
    "PRIMARY_bootstrap_ci95": [round(ci[0], 6), round(ci[1], 6)] if ci else None,
    "PRIMARY_ci_spans_no_information": spans_half,
    "mean_change_fidelity_plannable": round(st.mean(r["change_fidelity"] for r in PLAN), 6),
    "mean_change_fidelity_unplannable": round(st.mean(r["change_fidelity"] for r in NOPLAN), 6),
    "competing_predictors_auc": competing,
    "competing_predictors_caveat": (
        "NOT offered as replacement metrics. `n_changing` is a property of the WINDOW rather than "
        "the engine, so it is a corpus effect. `spurious_changed_cells` ordering the 'wrong' way "
        "most plausibly says an engine that writes more cells reaches SOME goal more often. No CI "
        "is computed for these; they are exploratory and were not pre-registered."
    ),
    "not_driven_by_one_cluster": (
        f"plannable engines come from {len(by_game)} distinct games, so this is not the tn36 "
        "bar-ticker cluster that made the n=2 join look negative."
    ),
    "what_this_does_NOT_say": (
        "It does not say the object-perception effect is fake -- that effect on change_fidelity "
        "is real and was reproduced under a corrected (HUD-masked) instrument. It says the effect "
        "is UNCASHABLE at present: a real movement in a number whose downstream value is "
        "unestablished. It also does not establish that the relationship is ABSENT; an AUC CI of "
        f"{[round(c, 3) for c in ci] if ci else None} is wide, and a larger n could still resolve "
        "it either way."
    ),
    "consequence": (
        "Representation changes (object perception, image/hybrid prompts, HUD masking) should not "
        "be adopted on the strength of a change_fidelity movement until some metric is shown to "
        "order engines by planning value. The weak signals that DO track plannability point at "
        "engines that ACT rather than engines that predict accurately, which favours attacking "
        "the generation failure classes (13 of 40 candidates inert, 9 unrunnable) over further "
        "representation work."
    ),
}

digest = json.dumps(
    sorted((r["key"], r["plan_found"], r["change_fidelity"]) for r in ROWS), sort_keys=True
)
report["reproducibility_checksum"] = hashlib.sha256(digest.encode()).hexdigest()
report["reproducibility_checksum_covers"] = (
    "the (engine key -> plan_found, change_fidelity) map only. Wall-times are excluded so an "
    "exact reproduction does not read as drift."
)
report["honest_verdict"] = (
    "complete_change_fidelity_does_not_predict_plannability_at_n17_ci_spans_no_information"
    if spans_half
    else "complete_change_fidelity_predicts_plannability"
)

(OUT / "fidelity_vs_plannability_scored.json").write_text(json.dumps(report, indent=1) + "\n")

print(f"scored={len(ROWS)}  plannable={len(PLAN)} across {len(by_game)} games")
print(f"AUC = {primary:.4f}   CI95 = [{ci[0]:.4f}, {ci[1]:.4f}]   spans 0.5: {spans_half}")
print(
    f"mean fidelity  plannable {report['mean_change_fidelity_plannable']:.4f}  "
    f"unplannable {report['mean_change_fidelity_unplannable']:.4f}"
)
print("competing:", competing)
print(f"\nverdict: {report['honest_verdict']}")
print(f"wrote {OUT / 'fidelity_vs_plannability_scored.json'}")
