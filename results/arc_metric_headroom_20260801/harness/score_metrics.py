#!/usr/bin/env python3
"""METRIC-HEADROOM, STEP 1 -- re-score the 48 FROZEN best-of-N induction candidates under every
candidate metric, and report which (if any) has headroom.

THE QUESTION, and why it blocks everything downstream.

`experiment_6018` A/B'd object perception on the induction prompt and returned

    complete_object_perception_heldout_ab_unmeasurable_instrument_floor_primary_zero_both_arms
    _no_test_possible_zero_discordant_pairs_n_support_games_14

because its pre-registered primary -- held-out exact-full-grid transition accuracy -- was exactly
0.0 in BOTH arms on all 168 cells. Zero discordant per-game pairs means no sign test exists; the
minimum reachable two-sided p is 1.0 no matter what the treatment does. That is NOT a null about
object perception. It is an instrument floor, and it is the FALSE_NEGATIVE_RISK pattern CLAUDE.md
names: "the method does not help" and "the measurement had no headroom" are indistinguishable.

So no representation change -- object graph, image, anything -- can be evaluated until a graded
metric with headroom exists. This run looks for one, and is explicitly willing to report that
none exists.

WHY THIS CORPUS. The 48 candidates of `results/arc_induce_bestofn_20260731` are 6 games x 8
samples, generated 2026-07-31 and frozen: completions on disk, transitions on disk, and a split
that was PROVEN row-by-row against the prompt TEXT (`harness/split.py`) rather than asserted. So
the held-out set really is unseen, and re-scoring it needs no GPU and no generation -- the
measurement is arithmetic over recorded transitions. It is also an INDEPENDENT corpus from the
one that floored: exp6018 measured 14 games x 6 replicates x 2 arms with its own generations.
Two corpora agreeing is worth more than one corpus re-analysed.

WHAT "HEADROOM" MEANS HERE, made concrete rather than adjectival. Four things are reported per
metric, and a metric must clear all four to be recommendable:

  1. `distinct_values` -- how many DIFFERENT values it takes across the candidates. A metric with
     one value cannot test anything, which is precisely what happened to exact-match.
  2. `is_floored` -- every measured value identical (and, in the hard case, identically 0.0).
  3. `n_discordant_pairs_available` -- THE DECIDING NUMBER. Over candidate pairs WITHIN a game
     (the only paired comparison that is not confounded by which game it is), how many pairs the
     metric assigns different values to. Under exact-match-on-changing-rows this collapses, which
     is why exp6018 could not run a test. Reported alongside `n_gradable_games`, which is the
     ceiling on the sign test an A/B could actually run.
  4. `spearman_vs_exact_match` -- rank agreement with exact-match ON THE SUBSET WHERE EXACT-MATCH
     IS NON-DEGENERATE. This is the guard against solving the problem by measuring something
     else: a graded metric that DISAGREES with exact-match exactly where exact-match works is not
     a graded version of it. Computed per game (exact-match varies within only some games) and
     pooled after within-game centring, because pooling raw values across games would measure
     which game is easier.

NO GENERATED CODE RUNS IN THIS INTERPRETER. Every engine invocation happens in `metric_worker.py`
inside a subprocess this driver can kill -- the pattern shipped as
`arc_engine_static_validation.dry_run_defects` after a non-terminating induced engine (ft09
candidate 5, which is in this very candidate set) wedged the generation loop for 13 minutes on
2026-07-31. A candidate that times out is recorded as UNDETERMINED and leaves every numerator and
denominator, rather than being scored 0 -- scoring a missing observation as a failure is the
error this project's artifact discipline is named for.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import pathlib
import pickle
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE.parent
BON = REPO / "results" / "arc_induce_bestofn_20260731"
BON_HARNESS = BON / "harness"

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_metric_headroom/e3")
# THE SHIPPED PROMPT DEFAULT MOVED UNDER THIS FROZEN CORPUS, ON THE DAY OF THIS RE-ANALYSIS.
# `_induce_transitions_k()` returned 8 when the 48 candidates were generated (2026-07-31); commit
# 253e1b60ed changed its default to None ("show ALL transitions") on 2026-08-01. `split.py`
# derives `shown` by replicating `changed[:k-2] + noop[:2]` with the CURRENT resolver, so an
# unpinned re-analysis is scoring against a split the frozen prompts never had. It happens to
# crash rather than silently mis-split (`None - 2` raises), but relying on that is luck, not a
# guard. The resolver's own docstring states that 8 "restores the previous prompt byte-for-byte",
# so it is pinned here and then VERIFIED field-by-field against the frozen split.json below --
# the pin is the intent, the verification is the evidence.
os.environ.setdefault("CARNOT_ARC_INDUCE_TRANSITIONS_K", "8")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(BON_HARNESS))  # reuse split.py's PROVEN split, never a second definition

TAGS = [t for t in os.environ.get("BON_TAGS", "gpu1").split(",") if t]
CALL_INDEX = int(os.environ.get("BON_CALL_INDEX", "1"))
WORKER_TIMEOUT_S = int(os.environ.get("MH_WORKER_TIMEOUT_S", "300"))
WORKERS = int(os.environ.get("MH_WORKERS", "6"))
SCRATCH = pathlib.Path(os.environ.get("MH_SCRATCH", "/tmp/arc_metric_headroom/code"))
SCRATCH.mkdir(parents=True, exist_ok=True)
PY = sys.executable
SEED = 20260801

# The metrics under test. (key, direction, family, one-line definition).
# `direction` is +1 where higher is better and -1 where lower is better; it matters only for how
# a delta would be READ, never for whether the metric has headroom.
METRICS: list[tuple[str, int, str, str]] = [
    (
        "exact_match_accuracy",
        +1,
        "transition_exact",
        "fraction of ALL held-out transitions whose predicted next grid equals the observed one "
        "exactly (WorldModelVerifier.accuracy) -- THE CONTROL, and exp6018's floored primary",
    ),
    (
        "change_exact_accuracy",
        +1,
        "transition_exact",
        "fraction of the CHANGING held-out transitions predicted exactly "
        "(WorldModelVerifier.change_accuracy); None where no changing row is held out",
    ),
    (
        "cell_recall",
        +1,
        "cell",
        "mean over changing transitions of the fraction of truly-changed cells given the right "
        "value (WorldModelVerifier.cell_recall); documented as BLIND to spurious writes",
    ),
    (
        "change_fidelity",
        +1,
        "cell",
        "mean over changing transitions of the fraction of the UNION of (cells reality "
        "changed) and (cells the engine wrote) that the engine got right -- symmetric, so "
        "a spurious write costs what a miss costs (WorldModelVerifier.change_fidelity)",
    ),
    (
        "correct_changed_cells",
        +1,
        "cell",
        "COUNT of truly-changed cells predicted correctly (unnormalised, so it scales with "
        "how much the game moves; kept because exp6018 tested it)",
    ),
    (
        "spurious_changed_cells",
        -1,
        "cell",
        "COUNT of cells the engine wrote that are not the observed value; minimised "
        "trivially by an engine that writes nothing, which is why it is reported and not "
        "recommended",
    ),
    (
        "grid_agreement_all",
        +1,
        "grid_distance",
        "1 - normalised Hamming distance between predicted and observed next grid, over ALL "
        "held-out rows",
    ),
    (
        "grid_agreement_changing",
        +1,
        "grid_distance",
        "1 - normalised Hamming distance, over CHANGING held-out rows only",
    ),
    (
        "changed_cell_jaccard",
        +1,
        "change_set",
        "Jaccard between the cell-set the engine WROTE and the cell-set reality CHANGED, ignoring "
        "the values -- 'did you change the right cells'",
    ),
    (
        "object_match_iou",
        +1,
        "object",
        "mean over changing transitions of the pixel-count-weighted F1 of best same-colour "
        "object-to-object pixel-IoU between the predicted and observed next grid -- GRADED "
        "per-object correctness (an object off by one pixel scores ~0.97, not 0)",
    ),
    (
        "object_match_recall",
        +1,
        "object",
        "the recall half of object_match_iou (true objects matched), blind to invented objects",
    ),
    (
        "object_inventory_jaccard",
        +1,
        "object",
        "multiset Jaccard over TRANSLATION-INVARIANT object hashes -- 'did you get the object "
        "inventory right', ignoring where each object ended up",
    ),
    (
        "object_positional_jaccard",
        +1,
        "object",
        "multiset Jaccard over (object hash, bbox top-left) -- object inventory AND position, "
        "all-or-nothing per object",
    ),
]
METRIC_KEYS = [m[0] for m in METRICS]


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rho with average ranks for ties.

    None when either side is constant: rho is undefined there, and reporting 0.0 would read as
    'the two metrics disagree' when the truth is 'this comparison carries no information'.
    """
    n = len(xs)
    if n < 3:
        return None

    def _rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return round(num / (dx * dy), 6) if dx > 0 and dy > 0 else None


def _run_workers(jobs: list[tuple[str, str]]) -> dict[str, dict]:
    """Bounded-parallel subprocess pool with a hard external kill. See the module docstring: a
    kill from OUTSIDE is the only bound a generated engine's own `except Exception` cannot
    swallow."""
    results: dict[str, dict] = {}
    pending = list(jobs)
    running: list[tuple[str, subprocess.Popen, float]] = []
    while pending or running:
        while pending and len(running) < WORKERS:
            key, jp = pending.pop(0)
            running.append(
                (
                    key,
                    subprocess.Popen(
                        [PY, str(HERE / "metric_worker.py"), jp],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    ),
                    time.monotonic(),
                )
            )
        time.sleep(0.1)
        for item in list(running):
            key, proc, t0 = item
            if proc.poll() is None:
                if time.monotonic() - t0 > WORKER_TIMEOUT_S:
                    proc.kill()
                    proc.wait(timeout=30)
                    results[key] = {
                        "status": "worker_timeout",
                        "worker_timeout_s": WORKER_TIMEOUT_S,
                    }
                    running.remove(item)
                continue
            so, se = proc.communicate()
            running.remove(item)
            try:
                results[key] = json.loads((so or "").strip().splitlines()[-1])
            except Exception:  # noqa: BLE001
                results[key] = {
                    "status": f"worker_rc{proc.returncode}",
                    "stderr": (se or "")[-300:],
                }
    return results


def main() -> int:  # noqa: C901
    t_start = time.monotonic()
    from carnot.agentic import arc_executable_world_model as e3
    from split import load_split

    runs = {}
    for tag in TAGS:
        p = BON_HARNESS / "bon" / tag / "bon.json"
        if p.exists():
            runs[tag] = json.loads(p.read_text())
    if not runs:
        print("no frozen best-of-N run found", file=sys.stderr)
        return 2

    games = sorted({g for r in runs.values() for g in r.get("games", [])})
    splits = {g: load_split(g, CALL_INDEX) for g in games}

    # SPLIT PROVENANCE GATE. Re-derive and check against the split PROVEN on 2026-07-31, field by
    # field, including the prompt sha. A mismatch means this re-analysis is grading a different
    # held-out set than the one the frozen numbers came from, which would make every comparison
    # below meaningless -- so it refuses rather than reporting. See the k-pin note at the top.
    frozen = {r["game"]: r for r in json.loads((BON / "split.json").read_text())["rows"]}
    split_check = []
    for g in games:
        s, f = splits[g], frozen.get(g, {})
        fields = (
            "n_full",
            "n_prefix",
            "n_shown",
            "n_heldout",
            "n_ambiguous_dropped",
            "heldout_n_changing",
            "heldout_n_noop",
            "prompt_sha256_16",
        )
        split_check.append(
            {
                "game": g,
                "matches_frozen_split_json": all(s.get(k) == f.get(k) for k in fields),
                "split_proven": bool(s["split_proven"]),
                "checks": dict(s["checks"]),
                "n_heldout": s["n_heldout"],
                "heldout_n_changing": s["heldout_n_changing"],
                "heldout_can_grade_change": bool(s["heldout_can_grade_change"]),
            }
        )
    if not all(r["matches_frozen_split_json"] and r["split_proven"] for r in split_check):
        print("SPLIT PROVENANCE GATE FAILED:", json.dumps(split_check, indent=1), file=sys.stderr)
        return 3
    caps = {
        g: json.loads((BON_HARNESS / "capture" / g / "capture.json").read_text())
        for g in games
        if (BON_HARNESS / "capture" / g / "capture.json").exists()
    }
    # Same measured partition score_bon.py uses: a post-bank induction at transition_count=1 is
    # near-vacuous and would flatter any metric, so the headline is the STALL path only.
    stall_games = sorted(g for g in games if int(caps.get(g, {}).get("levels_gained") or 0) == 0)

    rows: list[dict] = []
    jobs: list[tuple[str, str]] = []
    for tag, run in runs.items():
        for r in run.get("rows", []):
            game = r["game"]
            m = {
                "tag": tag,
                "game": game,
                "candidate": r["candidate"],
                "seed": r.get("seed"),
                "temperature": r.get("temperature"),
                "code_sha256_16": r.get("code_sha256_16"),
                "usable_at_generation": r.get("usable"),
                "is_stall_game": game in stall_games,
            }
            if r.get("status") != "ok":
                m["status"] = "generation_failed"
                rows.append(m)
                continue
            text = (BON_HARNESS / "bon" / tag / r["completion_file"]).read_text(errors="replace")
            code = e3._extract_python(text) or text.strip()
            cp = SCRATCH / f"{game}_k{r['candidate']}.py"
            cp.write_text(code)
            hp = SCRATCH / f"{game}_heldout.pkl"
            if not hp.exists():
                with open(hp, "wb") as fh:
                    pickle.dump(splits[game]["_heldout"], fh)
            jp = SCRATCH / f"{game}_k{r['candidate']}.job.json"
            jp.write_text(json.dumps({"code_path": str(cp), "heldout_pkl": str(hp)}))
            key = f"{game}|{r['candidate']}"
            m["_key"] = key
            jobs.append((key, str(jp)))
            rows.append(m)

    print(f"scoring {len(jobs)} candidates, {WORKERS} at a time ...", flush=True)
    res = _run_workers(jobs)
    for m in rows:
        k = m.pop("_key", None)
        if k and k in res:
            d = dict(res[k])
            m["status"] = d.pop("status", "ok")
            m.update(d)
        elif "status" not in m:
            m["status"] = "no_result"

    # A TIMEOUT IS A MISSING OBSERVATION, NOT A ZERO. `unrunnable:*` means no engine exists and
    # every metric is a genuine failure; `worker_timeout` means an engine exists and nothing about
    # it was learned. Folding the second into the denominator as a 0 records a missing observation
    # as a criterion failure -- see score_bon.py's `why_undetermined_is_not_a_zero`.
    def undetermined(m: dict) -> bool:
        return str(m.get("status", "")).startswith(("worker_timeout", "worker_rc", "no_result"))

    def value(m: dict, key: str):
        if undetermined(m):
            return None
        if m.get("status") != "ok":
            # No engine was produced at all: a genuine 0 on every similarity metric, and for the
            # two COUNT metrics the honest reading is likewise "nothing correct / nothing written".
            return 0.0 if key != "spurious_changed_cells" else 0.0
        return m.get(key)

    # ---- per-metric headroom analysis ------------------------------------------------------
    stall_rows = [m for m in rows if m["is_stall_game"]]
    analysis: dict[str, dict] = {}
    for key, direction, family, definition in METRICS:
        vals = [(m, value(m, key)) for m in stall_rows]
        measured = [(m, v) for m, v in vals if v is not None]
        nums = [round(float(v), 6) for _, v in measured]
        distinct = sorted(set(nums))
        n_und = sum(1 for m, v in vals if v is None and undetermined(m))
        n_unmeasurable = sum(1 for m, v in vals if v is None and not undetermined(m))

        # WITHIN-GAME candidate pairs: the only pairing not confounded by game difficulty.
        n_pairs_total = 0
        n_pairs_discordant = 0
        per_game: dict[str, dict] = {}
        for g in stall_games:
            gv = [round(float(v), 6) for m, v in measured if m["game"] == g]
            pairs = list(itertools.combinations(gv, 2))
            disc = sum(1 for a, b in pairs if a != b)
            n_pairs_total += len(pairs)
            n_pairs_discordant += disc
            per_game[g] = {
                "n_measured": len(gv),
                "n_distinct": len(set(gv)),
                "min": min(gv) if gv else None,
                "max": max(gv) if gv else None,
                "n_pairs": len(pairs),
                "n_discordant_pairs": disc,
                "gradable": len(set(gv)) > 1,
            }
        n_gradable_games = sum(1 for g in per_game.values() if g["gradable"])

        # Spearman vs the CONTROL, only where the control is non-degenerate.
        sp_per_game: dict[str, float | None] = {}
        pooled_x: list[float] = []
        pooled_y: list[float] = []
        for g in stall_games:
            xs, ys = [], []
            for m, v in measured:
                if m["game"] != g:
                    continue
                c = value(m, "exact_match_accuracy")
                if c is None:
                    continue
                xs.append(round(float(c), 6))
                ys.append(round(float(v), 6))
            if len(set(xs)) > 1 and len(xs) >= 3:
                sp_per_game[g] = _spearman(xs, ys)
                # Within-game CENTRING before pooling: pooling raw values across games would
                # measure which game is easier, not whether the two metrics agree.
                mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                pooled_x += [a - mx for a in xs]
                pooled_y += [b - my for b in ys]
        sp_pooled = _spearman(pooled_x, pooled_y) if len(pooled_x) >= 3 else None

        is_constant = len(distinct) <= 1
        analysis[key] = {
            "definition": definition,
            "family": family,
            "higher_is_better": direction > 0,
            "n_candidates_considered": len(vals),
            "n_measured": len(measured),
            "n_undetermined_excluded": n_und,
            "n_unmeasurable_excluded": n_unmeasurable,
            "distinct_values": len(distinct),
            "value_min": min(nums) if nums else None,
            "value_max": max(nums) if nums else None,
            "is_constant": is_constant,
            "is_hard_floored_all_zero": bool(nums) and all(v == 0.0 for v in nums),
            "separates_candidates": n_pairs_discordant > 0,
            "n_within_game_pairs": n_pairs_total,
            "n_discordant_pairs_available": n_pairs_discordant,
            "n_gradable_games": n_gradable_games,
            "n_stall_games": len(stall_games),
            # The floor on a 14-game paired sign test if EVERY gradable game were discordant --
            # exp6018's own `min_reachable_two_sided_p` formula, so the two are comparable.
            "min_reachable_two_sided_p_at_this_gradability": (
                round(min(1.0, 2 * 0.5**n_gradable_games), 10) if n_gradable_games else 1.0
            ),
            "spearman_vs_exact_match_per_game": sp_per_game,
            "spearman_vs_exact_match_pooled_within_game_centred": sp_pooled,
            "per_game": per_game,
        }

    payload = {
        "rows": rows,
        "metric_analysis": analysis,
        "stall_games": stall_games,
        "games": games,
        "n_candidates": len(rows),
        "random_seed": SEED,
        "call_index": CALL_INDEX,
        "worker_timeout_s": WORKER_TIMEOUT_S,
        "split_provenance": split_check,
        "induce_transitions_k_pinned": os.environ.get("CARNOT_ARC_INDUCE_TRANSITIONS_K"),
        "harness_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob("*.py"))
        },
        "duration_s": round(time.monotonic() - t_start, 3),
    }
    raw = OUT_DIR / "metric_scores_raw.json"
    raw.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {raw}")
    print(
        f"{'metric':34} {'nmeas':>5} {'distinct':>8} {'floor':>5} {'disc_pairs':>10} {'grad_g':>6}"
    )
    for key in METRIC_KEYS:
        a = analysis[key]
        print(
            f"{key:34} {a['n_measured']:>5} {a['distinct_values']:>8} "
            f"{str(a['is_hard_floored_all_zero']):>5} {a['n_discordant_pairs_available']:>10} "
            f"{a['n_gradable_games']:>6}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
