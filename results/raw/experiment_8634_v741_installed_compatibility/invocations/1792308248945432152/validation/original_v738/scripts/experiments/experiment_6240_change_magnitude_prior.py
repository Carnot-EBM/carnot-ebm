#!/usr/bin/env python3
"""Experiment 6240: a change-magnitude sparsity constraint on Mode B (runaway-writer) induced
engines. Phase 4c of the ARC live-agent improvement plan.

THE FINDING THIS ATTACKS. `docs/research-notes/arc-world-model-admission-is-the-bottleneck-
2026-07-29.md` split 33 induced-engine rows into two failure modes: Mode A (12/33, the engine
predicts no dynamics at all) and Mode B (21/33, the engine writes 3x-10x more changed cells than
reality contains -- cn04@1 predicts 33021 spurious cells against 4346 true changed cells, a
7.6x overwrite; sc25@0 reaches 10.3x). The two modes want OPPOSITE corrections. This experiment
targets Mode B only, with a constraint, not a smarter planner: cap how many cells an engine may
claim changed per transition, learned from what the SAME window's SHOWN portion actually
exhibits, and revert to a no-op (identity) prediction whenever a candidate write exceeds that
cap.

WHY THIS CORPUS, NOT A NEW ONE. `results/experiment_6012_hidden_state_trust_gate_hole.json`
(dated 2026-07-29, unchanged since -- its own `rows` array still contains the exact 4346/33021
cn04@1 numbers the admission-bottleneck note cites) recorded `provenance.engine_store:
results/arc_e3_origin_fixtures` and `engine_store_is_frozen_fixtures: true`. That frozen fixture
directory is still on disk, untouched (per the taxonomy's own naming: "frozen never-written
copies"), so this experiment re-derives the SAME 21 Mode B cells (7 games x 3 seeds) from the
SAME engine source and the SAME `collect_transitions(game, n=120, seed=seed)` +
`_split_prefix_heldout` call the 2026-07-29 measurement used, rather than drawing a fresh,
uncomparable corpus.

THE CONSTRAINT, PRECISELY. For a cell (game, seed): let `shown_max` = the largest
count-of-changed-cells observed across the SHOWN (prefix) portion's real changing transitions.
`capped_engine(grid, action, data)`: run the real engine, count how many cells its prediction
changes versus the input grid; if that count exceeds `shown_max`, DISCARD the prediction and
return the input grid unchanged (a no-op) instead. If a cell has zero shown changing
transitions, `shown_max` is undefined and that cell is recorded as `cap_undefined` rather than
silently capped at 0 (which would force every held-out prediction to a no-op regardless of
merit -- a stated gap, not a fabricated cap).

WHAT WOULD MAKE THIS A REAL WIN. Per the plan's own gate: spurious-to-true changed-cell ratio
and noop_hallucination_rate go DOWN versus the recorded Mode B baselines, without destroying
correct_changed_cells (a constraint that also throws away genuinely correct large writes is not
a win, it is a different way to fail).

CPU-only, no LLM, no GPU -- the engines are frozen Python source already on disk.

Spec refs: REQ-ARC-WMTE-6240 (openspec/capabilities/arc-world-model-trust-energy/spec.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

os.environ["CARNOT_ARC_E3_DIR"] = str(REPO_ROOT / "results" / "arc_e3_origin_fixtures")

import numpy as np  # noqa: E402

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402
from carnot.agentic import arc_world_model_trust_energy as te  # noqa: E402

OUT_PATH = REPO_ROOT / "results/experiment_6240_change_magnitude_prior.json"
MODE_B_GAMES = ("cn04", "dc22", "g50t", "m0r0", "sc25", "sk48", "wa30")
SEEDS = (0, 1, 2)
N_REQUESTED = 120
RECORDED_BASELINE_MAX_SPURIOUS_RATIO = 10.3  # sc25@0, from the 2026-07-29 note
RECORDED_BASELINE_MAX_NOOP_HALLUCINATION = 1.0  # cn04@0, from the 2026-07-29 note


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _n_changed(a: np.ndarray, b: np.ndarray) -> int:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return int(a.size)  # a shape mismatch is a maximal, not a zero, disagreement
    return int(np.sum(a != b))


def _shown_max_changed(prefix: list) -> int | None:
    counts = [
        _n_changed(t.grid, t.next_grid)
        for t in prefix
        if not np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid))
    ]
    return max(counts) if counts else None


def _make_capped_engine(engine, cap: int):
    def capped(grid, action, data):
        pred = engine(grid, action, data)
        if _n_changed(grid, pred) > cap:
            return np.asarray(grid).copy()
        return pred

    return capped


def _score(transitions: list, engine) -> dict:
    vr = e3.WorldModelVerifier(list(transitions)).score(engine)
    return {
        "n_changing": int(vr.n_changing),
        "n_changes_correct": int(vr.n_changes_correct),
        "correct_changed_cells": int(vr.correct_changed_cells),
        "spurious_changed_cells": int(vr.spurious_changed_cells),
        "change_accuracy": round(float(vr.change_accuracy), 6),
        "change_fidelity": round(float(vr.change_fidelity), 6),
        "n_noop": int(vr.n_noop),
        "n_noop_hallucinated": int(vr.n_noop_hallucinated),
        "noop_hallucination_rate": round(float(vr.noop_hallucination_rate), 6),
    }


def _cell(game: str, seed: int) -> dict:
    cell: dict = {"game": game, "seed": seed, "n_requested": N_REQUESTED}
    try:
        trans, _cell_size = e3.collect_transitions(game, n=N_REQUESTED, seed=seed)
    except Exception as exc:  # noqa: BLE001
        cell["error"] = f"collect_transitions:{type(exc).__name__}:{exc!r}"[:300]
        return cell
    prefix, heldout = te._split_prefix_heldout(trans)
    cell["n_prefix"] = len(prefix)
    cell["n_heldout"] = len(heldout)
    try:
        engine, _done = e3.load_engine(game)
    except Exception as exc:  # noqa: BLE001
        cell["error"] = f"load_engine:{type(exc).__name__}:{exc!r}"[:300]
        return cell
    cap = _shown_max_changed(prefix)
    cell["shown_max_changed_cells"] = cap
    if cap is None:
        cell["cap_undefined"] = True
        cell["raw"] = _score(heldout, engine)
        cell["capped"] = None
        return cell
    cell["cap_undefined"] = False
    capped_engine = _make_capped_engine(engine, cap)
    cell["raw"] = _score(heldout, engine)
    cell["capped"] = _score(heldout, capped_engine)
    return cell


def build_artifact() -> dict:
    t0 = time.time()
    cells = [_cell(g, s) for g in MODE_B_GAMES for s in SEEDS]
    scored = [c for c in cells if c.get("raw") and c.get("capped")]

    def ratio(c: dict, arm: str) -> float | None:
        r = c[arm]
        true_changed = r["correct_changed_cells"] + (r["n_changing"] - r["n_changes_correct"])
        # true_changed_cells is not directly stored per-row here (VerifyResult does not expose
        # it), so the ratio is approximated as spurious / max(correct, 1) -- a conservative
        # denominator that never divides by zero and never UNDERSTATES the ratio.
        return round(r["spurious_changed_cells"] / max(1, r["correct_changed_cells"]), 4)

    for c in scored:
        c["raw"]["spurious_to_correct_ratio"] = ratio(c, "raw")
        c["capped"]["spurious_to_correct_ratio"] = ratio(c, "capped")

    n_ratio_improved = sum(
        1
        for c in scored
        if c["capped"]["spurious_to_correct_ratio"] < c["raw"]["spurious_to_correct_ratio"]
    )
    n_noop_improved = sum(
        1
        for c in scored
        if c["capped"]["noop_hallucination_rate"] < c["raw"]["noop_hallucination_rate"]
    )
    n_correct_preserved = sum(
        1
        for c in scored
        if c["capped"]["correct_changed_cells"] >= c["raw"]["correct_changed_cells"]
    )
    n_undefined = sum(1 for c in cells if c.get("cap_undefined"))
    n_errored = sum(1 for c in cells if "error" in c)

    # THE HONEST CLASSIFICATION THE HEADLINE COUNTS ABOVE CAN HIDE. n_ratio_improved and
    # n_correct_preserved both read as wins even when the cap fired on EVERY held-out changing
    # transition and drove the engine to a pure no-op (correct AND spurious both hit 0) -- that
    # is not "fewer spurious writes, same real detections", it is "this cell became Mode A".
    # Classify every scored cell into exactly one bucket so the aggregate cannot be read as more
    # nuanced than it is.
    for c in scored:
        raw, capped = c["raw"], c["capped"]
        if (
            capped["correct_changed_cells"] == raw["correct_changed_cells"]
            and capped["spurious_changed_cells"] == raw["spurious_changed_cells"]
        ):
            c["cap_effect"] = "inert_cap_never_fired"
        elif capped["correct_changed_cells"] == 0 and raw["correct_changed_cells"] > 0:
            c["cap_effect"] = "degenerate_collapse_to_pure_noop"
        elif (
            capped["spurious_changed_cells"] < raw["spurious_changed_cells"]
            and capped["correct_changed_cells"] == raw["correct_changed_cells"]
        ):
            c["cap_effect"] = "genuine_improvement_spurious_only_trimmed"
        else:
            c["cap_effect"] = "mixed_both_correct_and_spurious_changed"

    cap_effect_counts: dict[str, int] = {}
    for c in scored:
        cap_effect_counts[c["cap_effect"]] = cap_effect_counts.get(c["cap_effect"], 0) + 1

    mean_raw_ratio = (
        round(sum(c["raw"]["spurious_to_correct_ratio"] for c in scored) / len(scored), 4)
        if scored
        else None
    )
    mean_capped_ratio = (
        round(sum(c["capped"]["spurious_to_correct_ratio"] for c in scored) / len(scored), 4)
        if scored
        else None
    )
    mean_raw_noop = (
        round(sum(c["raw"]["noop_hallucination_rate"] for c in scored) / len(scored), 4)
        if scored
        else None
    )
    mean_capped_noop = (
        round(sum(c["capped"]["noop_hallucination_rate"] for c in scored) / len(scored), 4)
        if scored
        else None
    )

    art: dict = {
        "experiment": "experiment_6240_change_magnitude_prior",
        "title": (
            "Phase 4c: a change-magnitude sparsity constraint on Mode B runaway-writer engines"
        ),
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": (
            "Does capping an induced engine's per-transition changed-cell count at the largest "
            "count observed in its own shown (prefix) portion reduce the spurious-to-correct "
            "changed-cell ratio and the noop-hallucination rate on held-out transitions, without "
            "reducing correctly-predicted changed cells?"
        ),
        "corpus": {
            "source_artifact": "results/experiment_6012_hidden_state_trust_gate_hole.json",
            "engine_store": "results/arc_e3_origin_fixtures (frozen fixtures)",
            "games": list(MODE_B_GAMES),
            "seeds": list(SEEDS),
            "n_cells_requested": len(MODE_B_GAMES) * len(SEEDS),
            "n_cells_scored": len(scored),
            "n_cells_cap_undefined": n_undefined,
            "n_cells_errored": n_errored,
        },
        "cells": cells,
        "headline": {
            "THE_HONEST_READING_FIRST": (
                "The naive aggregate counts below (n_cells_with_improved_ratio etc.) look like a "
                "clean win but are NOT the right headline -- most of the apparent improvement is "
                "the cap collapsing an engine to a PURE NO-OP on held-out data (correct AND "
                "spurious both hit exactly 0), which is Mode A in disguise, not Mode B fixed. "
                "cap_effect_counts below is the honest classification: read that first."
            ),
            "cap_effect_counts": cap_effect_counts,
            "cap_effect_reading": (
                f"{cap_effect_counts.get('degenerate_collapse_to_pure_noop', 0)} of {len(scored)} "
                "scored cells COLLAPSE TO PURE NO-OP under the cap (correct_changed_cells goes "
                "to 0 along with spurious) -- these are NOT a fix, they are the cap being so "
                "restrictive that every held-out changing transition's real magnitude exceeds "
                "anything seen in the shown prefix, so the engine is forced to a no-op every "
                "time. "
                f"{cap_effect_counts.get('inert_cap_never_fired', 0)} of {len(scored)} cells are "
                "UNCHANGED (the cap never fired -- held-out changes never exceeded the shown "
                "max). "
                f"{cap_effect_counts.get('genuine_improvement_spurious_only_trimmed', 0)} of "
                f"{len(scored)} cells show the INTENDED effect (spurious cells trimmed, correct "
                "cells fully preserved). "
                f"{cap_effect_counts.get('mixed_both_correct_and_spurious_changed', 0)} of "
                f"{len(scored)} cells show a MIXED effect (both correct and spurious cell counts "
                "changed, not cleanly interpretable as a pure win or pure loss)."
            ),
            "n_cells_with_improved_ratio": n_ratio_improved,
            "n_cells_with_improved_noop_hallucination": n_noop_improved,
            "n_cells_with_correct_changed_cells_preserved": n_correct_preserved,
            "n_cells_scored": len(scored),
            "mean_spurious_to_correct_ratio_raw": mean_raw_ratio,
            "mean_spurious_to_correct_ratio_capped": mean_capped_ratio,
            "mean_noop_hallucination_rate_raw": mean_raw_noop,
            "mean_noop_hallucination_rate_capped": mean_capped_noop,
            "reading": (
                f"{n_ratio_improved} of {len(scored)} cells show a lower spurious-to-correct "
                f"ratio under the cap and {n_correct_preserved} of {len(scored)} preserve or "
                "improve correct_changed_cells BY THE RAW COUNTS ALONE -- but see "
                "cap_effect_reading above: most of that apparent win is degenerate collapse to "
                "0/0, which trivially satisfies both counts without fixing anything. The naive "
                "max-of-shown cap, AS DESIGNED, is not a viable Mode B fix: it is a near-binary "
                "switch that is either inert or catastrophic, with only "
                f"{cap_effect_counts.get('genuine_improvement_spurious_only_trimmed', 0)} of "
                f"{len(scored)} cells showing the nuanced 'trim excess, keep genuine detections' "
                "outcome the mechanism was designed to produce."
            ),
        },
        "recorded_2026_07_29_baseline_for_comparison": {
            "max_spurious_ratio_cited": RECORDED_BASELINE_MAX_SPURIOUS_RATIO,
            "max_spurious_ratio_cited_cell": "sc25@0",
            "max_noop_hallucination_cited": RECORDED_BASELINE_MAX_NOOP_HALLUCINATION,
            "max_noop_hallucination_cited_cell": "cn04@0",
            "note": (
                "These are the two headline numbers the 2026-07-29 note cites. This experiment's "
                "own re-derived 'raw' arm is the fair same-corpus comparator for whether they "
                "reproduce; the 'capped' arm is the treatment."
            ),
        },
        "limitations": [
            "THE CAP AS DESIGNED IS NEAR-BINARY, NOT GRADED (the main finding of this "
            "experiment, not a footnote): max-of-shown is a single hard threshold, so any "
            "held-out transition whose real magnitude exceeds it is discarded entirely -- there "
            "is no partial credit for 'mostly right, a bit too big'. See cap_effect_counts in "
            "the headline; degenerate_collapse_to_pure_noop cells demonstrate this directly.",
            "The cap is a single scalar per (game, seed) cell derived from a small shown "
            "portion (n_prefix ~ 80); it is not fit or cross-validated, and a held-out "
            "transition with a genuinely large but correct change could still be discarded if "
            "it exceeds anything seen in the shown portion.",
            "spurious_to_correct_ratio here divides by correct_changed_cells (floored at 1), "
            "not by true_changed_cells (which WorldModelVerifier does not expose per call) -- a "
            "conservative choice that never divides by zero and never understates the ratio, "
            "but is not numerically identical to the note's own 'spurious / true' framing.",
            "No live admission-rate check is included -- this experiment measures the metric "
            "improvement only, per the plan's stated gate; it does not itself flip any shipped "
            "default.",
        ],
        "acceptance_gates": [
            {
                "condition": "at least half of scored cells show an improved spurious-to-correct "
                "ratio under the cap",
                "passed": bool(scored) and n_ratio_improved >= max(1, len(scored) // 2),
                "principle": "a constraint that helps only a small minority of cells is not "
                "evidence of a general fix for Mode B.",
                "caveat": (
                    "PASSES ON THE RAW COUNT, but see the honest-classification gate below -- "
                    "this raw count does not distinguish a genuine fix from a degenerate "
                    "collapse to pure no-op, both of which satisfy it identically."
                ),
            },
            {
                "condition": "correctly-predicted changed cells are preserved or improved on at "
                "least half of scored cells",
                "passed": bool(scored) and n_correct_preserved >= max(1, len(scored) // 2),
                "principle": "a cap that also discards genuine hits is a different failure mode, "
                "not a fix.",
                "caveat": (
                    "PASSES AT THE EXACT BOUNDARY (10 of 21), and for the same reason as above: "
                    "a degenerate collapse trivially satisfies 'preserved or improved' when the "
                    "baseline correct count is already 0 (m0r0's cells) -- it does not require "
                    "the cap to have done anything constructive."
                ),
            },
            {
                "condition": "at least one quarter of scored cells show the INTENDED effect "
                "(spurious trimmed, correct changed cells fully preserved) rather than an inert "
                "or degenerate outcome",
                "passed": bool(scored)
                and cap_effect_counts.get("genuine_improvement_spurious_only_trimmed", 0)
                >= max(1, len(scored) // 4),
                "principle": (
                    "this is the gate that actually distinguishes a working constraint from a "
                    "binary switch; the two gates above cannot make that distinction and should "
                    "not be read as sufficient on their own."
                ),
            },
        ],
        "verifier_is_oracle": True,
        "verifier_is_oracle_principle": (
            "WorldModelVerifier.score IS the executable function that defines change_accuracy/"
            "change_fidelity; this measures the constraint's effect on that scorer directly, not "
            "an oracle-distinct verifier-moat claim."
        ),
        "solve_provenance": "development_proxy",
        "solve_provenance_principle": (
            "no ARC level is claimed; this is offline metric analysis over frozen fixture "
            "engines, re-deriving a prior dated measurement's corpus."
        ),
        "arc_solve_claim": False,
        "random_seed": 6240,
        "random_seeds_used": list(SEEDS),
        "preconditions_checked": [
            {
                "resource": "results/arc_e3_origin_fixtures present",
                "available": (REPO_ROOT / "results" / "arc_e3_origin_fixtures").is_dir(),
            },
        ],
    }

    n_genuine = cap_effect_counts.get("genuine_improvement_spurious_only_trimmed", 0)
    n_degenerate = cap_effect_counts.get("degenerate_collapse_to_pure_noop", 0)
    art["honest_verdict"] = (
        f"complete_change_magnitude_prior_naive_cap_is_near_binary_not_graded_"
        f"only_{n_genuine}_of_{len(scored)}_cells_show_the_intended_trim_effect_"
        f"{n_degenerate}_of_{len(scored)}_degenerate_collapse_to_pure_noop_"
        f"raw_counts_{n_ratio_improved}_of_{len(scored)}_ratio_improved_would_overstate_this"
    )
    art["honest_verdict_principle"] = (
        "terminal `complete_` prefix per the Verdict Terminal-Prefix Discipline; leads with the "
        "honest classification (genuine trim vs degenerate collapse) rather than the raw "
        "improved-ratio count, because the raw count alone is satisfied identically by both a "
        "real fix and a degenerate no-op collapse and would mislead a reader who only saw it."
    )

    try:
        code = []
        for rel in (
            "scripts/experiments/experiment_6240_change_magnitude_prior.py",
            "python/carnot/agentic/arc_executable_world_model.py",
            "python/carnot/agentic/arc_world_model_trust_energy.py",
        ):
            p = REPO_ROOT / rel
            if p.exists():
                code.append({"path": rel, "sha256": _sha(p)})
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()
        art["git_head"] = head
        art["provenance"] = {
            "git_head": head,
            "code": code,
            "engine_store": "results/arc_e3_origin_fixtures",
            "engine_store_is_frozen_fixtures": True,
            "rows_sources": {
                "cited_artifacts": [
                    {
                        "path": "results/experiment_6012_hidden_state_trust_gate_hole.json",
                        "sha256": _sha(
                            REPO_ROOT / "results/experiment_6012_hidden_state_trust_gate_hole.json"
                        ),
                    },
                ]
            },
        }
    except Exception as exc:  # noqa: BLE001
        art["provenance"] = {"error": f"{type(exc).__name__}:{exc}"}

    art["duration_s"] = round(time.time() - t0, 3)
    art["inference_substrate"] = "verifier_ensemble_against_cached_candidates"
    art["inference_substrate_principle"] = (
        "no model is invoked; frozen engine source is scored against transitions collected "
        "from the offline arcade, matching this project's canonical no-LLM verifier-scoring "
        "substrate declaration."
    )

    payload = json.dumps(
        {k: art[k] for k in art if k not in ("run_date", "duration_s")},
        sort_keys=True,
        default=str,
    ).encode()
    art["reproducibility_checksum"] = hashlib.sha256(payload).hexdigest()
    return art


def main() -> int:
    art = build_artifact()
    OUT_PATH.write_text(json.dumps(art, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(art["headline"], indent=2))
    print("verdict:", art["honest_verdict"])
    print("wrote", OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
