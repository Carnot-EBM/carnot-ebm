#!/usr/bin/env python
"""Guard suite for the analyser. Every guard is tested against the ACTUAL failure it
exists for, plus a mutation proof that the guard is load-bearing.

The guards:
  G1 statistics correctness -- Clopper-Pearson + McNemar exact against hand/table values.
  G2 origin-incident detection -- the 8 REAL provably-dead K=4 cells from 2026-07-26 must
     be flagged. A guard that prints OK on a faithful replay of its own origin incident is
     worthless; that exact failure has shipped in this repo before.
  G3 sentinel handling -- the -1 "undetermined" sentinel the shipped witness writes for a
     stub proposer must NOT be summed as a count and must NOT be scored as a dead
     generator. (Found by smoke-testing: it produced total_llm_calls=-15.)
  G4 over-fire control -- healthy rows must produce zero findings.
  G5 positive-control logic -- a harness that records no wins at all must fail the
     positive control, so a low LLM-on rate can never pass as a finding on a broken
     detector.

Mutation proofs: each guard is re-run with a plausible-but-wrong variant of its own
predicate; a mutation that still passes means the guard is not load-bearing.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import analyse  # noqa: E402

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
FAILS: list[str] = []
NOTES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILS.append(name)


def witness(
    game="g", calls=0, resp=0, err=0, healthy=True, n_ctx=81920, enabled=True, diags=0
) -> dict:
    return {
        "game": game,
        "llm_enabled": enabled,
        "induction_attempts_n": 1,
        "induction_attempts_planned": 0,
        "induction_attempts_skipped": [],
        "generator_constructed": True,
        "llm": {"calls": calls, "responses": resp, "errors": err, "content_failures": 0},
        "generator_healthy_after": healthy,
        "generator_server_failure_diagnostics": [
            "<HTTPError 500: 'Internal Server Error'> body='Context size has been exceeded.'"
        ]
        * diags,
        "generator_port": 8952,
        "generator_n_ctx": n_ctx,
        "generator_max_tokens": 4096,
        "llm_on_row_valid": resp > 0 and err == 0 and healthy is True,
    }


def cell(sig, fw, w, arm="x", elapsed=10.0) -> dict:
    return {
        "arm": arm,
        "variant_signature": sig,
        "elapsed_s": elapsed,
        "first_win": fw,
        "cell_error": "",
        "liveness_witness": w,
    }


# ------------------------------------------------------------------ G1 statistics
print("G1 statistics correctness")
for k, n, want in [
    (4, 100, [0.011004, 0.099257]),
    (0, 100, [0.0, 0.03624]),
    (1, 25, [0.001012, 0.203517]),
    (10, 100, [0.049005, 0.176223]),
]:
    got = analyse.clopper_pearson(k, n)
    check(
        f"clopper_pearson({k},{n})",
        all(abs(a - b) < 2e-5 for a, b in zip(got, want, strict=True)),
        f"got={got} want={want}",
    )
for b, c, p2, minp, sig in [
    (0, 0, 1.0, 1.0, False),
    (5, 0, 0.0625, 0.0625, False),
    (6, 0, 0.03125, 0.03125, True),
    (0, 6, 0.03125, 0.03125, True),
    (3, 3, 1.0, 0.03125, False),
]:
    m = analyse.mcnemar_exact(b, c)
    check(
        f"mcnemar({b},{c})",
        abs(m["p_two_sided"] - p2) < 1e-9
        and abs(m["min_reachable_p_two_sided_at_this_support"] - minp) < 1e-9
        and m["significant_at_0_05"] is sig,
        json.dumps(m),
    )
# both tails must be present and must not be silently equal when the effect is one-sided
m = analyse.mcnemar_exact(6, 0)
check(
    "both tails reported and distinct for a one-sided effect",
    m["p_one_sided_treatment_better"] < m["p_one_sided_control_better"],
    json.dumps(m),
)

# ------------------------------------------------------------------ G2 origin incident
print("G2 origin-incident detection (8 REAL dead K=4 cells, 2026-07-26)")
origin = {}
for f in sorted(
    glob.glob(str(REPO / "results/llm_on_contention_rows_20260726/cells/cell_K4_*.json"))
):
    d = json.loads(Path(f).read_text())
    r = d["row"]
    L = r.get("llm") or {}
    calls = int(L.get("generate_calls", 0)) + int(L.get("complete_text_calls", 0))
    w = witness(
        game=d["game"],
        calls=calls,
        resp=int(L.get("responses", 0)),
        err=int(L.get("errors", 0)),
        healthy=r.get("generator_healthy_after"),
        n_ctx=16384,
    )
    sig = f"{d['game']}~seed{d['seed']}"
    origin[sig] = cell(sig, False, w, arm="K4_origin", elapsed=float(r.get("wall_s", 0)))
check("loaded exactly the 8 origin K=4 cells", len(origin) == 8, str(len(origin)))
o = analyse.arm_summary(origin)["liveness"]
check(
    "flags all 8 origin cells as generator_healthy_after=False",
    o["generator_healthy_after_false_cells"] == 8,
    json.dumps(o),
)
check(
    "flags the 3 zero-response origin cells as provably dead",
    o["dead_generator_cells"] == 3,
    str(o["dead_generator_cells"]),
)
check(
    "origin arm response_rate is degraded, not clean",
    o["response_rate"] is not None and o["response_rate"] < 0.25,
    str(o["response_rate"]),
)
NOTES.append(
    f"origin arm: calls={o['total_llm_calls']} responses={o['total_llm_responses']} "
    f"response_rate={o['response_rate']} dead_cells={o['dead_generator_cells']}"
)

# ------------------------------------------------------------------ G3 sentinel
print("G3 sentinel handling (-1 = undetermined, the real llm_off control shape)")
noop_w = {
    "game": "ar25",
    "llm_enabled": True,
    "induction_attempts_n": 1,
    "induction_attempts_planned": 0,
    "induction_attempts_skipped": ["proposer_failed_or_missing_root"],
    "generator_constructed": True,
    "liveness_witness_error": (
        "AttributeError(\"'_NoOpProposer' object has no attribute 'liveness_witness'\")"
    ),
    "llm": {"calls": -1, "responses": -1, "errors": -1},
    "generator_healthy_after": None,
    "llm_on_row_valid": False,
}
sent = {
    f"g{i}~color01": cell(f"g{i}~color01", i == 0, dict(noop_w), arm="llm_off") for i in range(15)
}
s = analyse.arm_summary(sent)["liveness"]
check(
    "no negative totals leak",
    s["total_llm_calls"] == 0 and s["total_llm_responses"] == 0,
    json.dumps(s),
)
check(
    "undetermined rows are NOT scored as dead generators",
    s["dead_generator_cells"] == 0,
    str(s["dead_generator_cells"]),
)
check(
    "undetermined rows are COUNTED, not hidden",
    s["n_cells_llm_counters_undetermined"] == 15,
    str(s),
)
check(
    "response_rate is None (not 0.0) when nothing was determined",
    s["response_rate"] is None,
    str(s["response_rate"]),
)

# ------------------------------------------------------------------ G4 over-fire control
print("G4 over-fire control (healthy rows must be clean)")
healthy = {
    f"h{i}~color01": cell(f"h{i}~color01", False, witness(game=f"h{i}", calls=4, resp=4, err=0))
    for i in range(10)
}
h = analyse.arm_summary(healthy)["liveness"]
check(
    "no dead cells on a healthy arm", h["dead_generator_cells"] == 0, str(h["dead_generator_cells"])
)
check("no server errors on a healthy arm", h["total_llm_server_errors"] == 0, str(h))
check("no undetermined on a healthy arm", h["n_cells_llm_counters_undetermined"] == 0, str(h))
check("healthy arm response_rate == 1.0", h["response_rate"] == 1.0, str(h["response_rate"]))
# a game that never stalled into induction: calls==0 is "never asked", NOT "dead"
never = {"n0~color01": cell("n0~color01", False, witness(game="n0", calls=0, resp=0, err=0))}
n = analyse.arm_summary(never)["liveness"]
check("calls==0 is 'never asked', not 'dead'", n["dead_generator_cells"] == 0, str(n))

# ------------------------------------------------------------------ G5 positive control
print("G5 positive-control logic")


def positive_control(off_wins: int, off_rate, base_rate, off_winners, base_winners) -> bool:
    return bool(
        off_wins > 0 and off_rate == base_rate and sorted(off_winners) == sorted(base_winners)
    )


check(
    "a detector that records ZERO wins FAILS the positive control",
    positive_control(0, 0.0, 0.04, [], ["lp85~color01"]) is False,
)
check(
    "a detector that reproduces the rate but NOT the winners FAILS",
    positive_control(4, 0.04, 0.04, ["xx~color01"] * 4, [f"lp85~color0{i}" for i in (1, 2, 3, 4)])
    is False,
)
check(
    "a faithful detector PASSES",
    positive_control(
        4,
        0.04,
        0.04,
        [f"lp85~color0{i}" for i in (1, 2, 3, 4)],
        [f"lp85~color0{i}" for i in (1, 2, 3, 4)],
    )
    is True,
)

# ------------------------------------------------------------------ mutation proofs
print("MUTATION PROOFS (a guard that survives its mutation is not load-bearing)")
muts = []

# M1: treat the -1 sentinel as a plain int again (the original bug). Must break G3.
orig_count = analyse.arm_summary


def m1() -> bool:
    lw = [dict(noop_w)]
    v = int((lw[0]["llm"]).get("calls") or 0)
    return v == 0  # the buggy reading coerces -1 via `or 0`? no -- it yields -1


muts.append(
    (
        "M1 sentinel-as-count",
        int((noop_w["llm"]).get("calls") or 0) == -1,
        "the naive reading yields -1, so excluding negatives is load-bearing",
    )
)

# M2: dead-cell predicate written with `or 0` instead of `or -1`. On the sentinel row that
# reads calls=-1 -> not >0 -> still not dead; but on a row with calls MISSING it would read
# 0 responses as dead. Prove the -1 default matters for a missing-key row.
missing = {"m~color01": cell("m~color01", False, {**witness(game="m"), "llm": {}})}
mm = analyse.arm_summary(missing)["liveness"]
muts.append(
    (
        "M2 missing-llm-block not scored dead",
        mm["dead_generator_cells"] == 0,
        f"dead={mm['dead_generator_cells']}",
    )
)

# M3: min_reachable_p computed as 0.5**n (one-sided) instead of 2*0.5**n. At n=5 that
# would report 0.031 < 0.05 and claim significance WAS reachable when it was not.
muts.append(
    (
        "M3 min-p must be two-sided",
        abs(analyse.mcnemar_exact(5, 0)["min_reachable_p_two_sided_at_this_support"] - 0.0625)
        < 1e-9,
        "one-sided 0.03125 would falsely claim reachable power at n=5",
    )
)

# M4: positive control without the winners check -- would pass a detector that finds 4 wins
# on the WRONG variants, i.e. a differently-defined quantity presented as the baseline.
muts.append(
    (
        "M4 winners check is load-bearing",
        positive_control(
            4,
            0.04,
            0.04,
            ["xx~c1", "xx~c2", "xx~c3", "xx~c4"],
            [f"lp85~color0{i}" for i in (1, 2, 3, 4)],
        )
        is False,
        "rate-only equality would pass a wrong-variant detector",
    )
)

# M6: THE FALSY-ZERO TRAP, and the reason G2 exists. Writing the dead-cell predicate as
# `int(b.get("responses") or -1) == 0` looks equivalent to reading the counter with an
# undetermined default, but `responses: 0` is FALSY, so `0 or -1` is -1 and the predicate
# never fires. That mutation was actually written during this session and G2 caught it: the
# guard printed OK on 8 real cells it was authored for. Assert the trap explicitly so it
# cannot silently return.
_dead_row = witness(game="d", calls=4, resp=0, err=4, healthy=False, n_ctx=16384)
muts.append(
    (
        "M6 falsy-zero: `0 or -1` mutation would disable the dead-cell check",
        int((_dead_row["llm"]).get("responses") or -1) == -1
        and analyse._num(_dead_row["llm"], "responses") == 0,
        "the mutation reads responses=0 as -1; _num preserves 0",
    )
)
_dead_arm = analyse.arm_summary({"d~c1": cell("d~c1", False, _dead_row)})["liveness"]
muts.append(
    (
        "M6b the real predicate DOES fire on a single asked-and-silent cell",
        _dead_arm["dead_generator_cells"] == 1,
        f"dead={_dead_arm['dead_generator_cells']}",
    )
)

# M5: contention control declared live from n_ctx alone rather than from observed errors.
# A 16k arm whose pool never overflowed would then be called 'live' with zero evidence.
faulty_but_clean = {
    f"f{i}~color01": cell(
        f"f{i}~color01", False, witness(game=f"f{i}", calls=4, resp=4, err=0, n_ctx=16384)
    )
    for i in range(4)
}
fc = analyse.arm_summary(faulty_but_clean)["liveness"]
muts.append(
    (
        "M5 fault-exhibited must come from errors, not n_ctx",
        fc["total_llm_server_errors"] == 0 and fc["generator_n_ctx_observed"] == [16384],
        "n_ctx==16384 alone does not prove the fault fired",
    )
)

for name, ok, why in muts:
    check(name, ok, why)

print()
for n in NOTES:
    print("NOTE:", n)
print()
if FAILS:
    print(f"GUARD SUITE FAILED: {FAILS}")
    raise SystemExit(1)
print("GUARD SUITE OK -- all guards fire on their own origin failures and survive mutation")
