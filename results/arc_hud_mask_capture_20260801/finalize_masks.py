#!/usr/bin/env python3
"""Stage 2: apply the SECOND-CORPUS gate and write the FINAL per-game mask files.

PIPELINE ORDER (each stage consumes the previous stage's file, nothing is re-measured):

    capture.py            -> per_game.json           (detect + single-corpus audit;
                                                      writes PROVISIONAL mask files)
    cross_corpus_check.py -> cross_corpus_check.json (the random-action corpus + the
                                                      explorer's own mask, per game)
    finalize_masks.py     -> masks/*.json  FINAL     (this file; the second-corpus gate)
    build_artifact.py     -> hud_mask_capture.json   (the scored artifact)

WHY A SECOND STAGE AT ALL. The capture accepted a band on lf52 and su15 -- the exact two
games REQ-ARC-WMTE-6015 documents as over-masking cases. Re-measured here on the corpus
that table was taken from (`collect_transitions(n=60, seed=0)`), the numbers reproduce that
table to four decimals:

    lf52  overlap 1.0000, 60 changing -> 0 surviving   (table: 1.0000, 60 -> 0)
    su15  overlap 0.7568, 28 changing -> 1 surviving   (table: 0.7568, 28 -> 1)
    s5i5  overlap 0.2219  (table: "every other game 0.0000 .. 0.2219, s5i5 is the highest")

So the hazard is real, it is reproduced, and a single-corpus accept is not enough to ship a
mask. REQ-ARC-WMTE-6017 already stated the governing rule -- "a verdict is a statement about
(mask, corpus), never about the mask alone" -- and recorded lf52's verdict FLIPPING between
a random-action corpus (refused) and a live episode (applied). Both records are honest; what
is NOT honest is shipping the accept and not mentioning the refusal.

THE GATE. A band reaches `hud_rows`/`hud_cols` only if it is affirmatively clean on BOTH
corpora. A band clean on the winning route but refused on the random corpus is WITHHELD --
moved to `conditionally_clean_bands`, which is NOT part of the mask a consumer materialises.
It is kept rather than deleted for two reasons: the project's never-prune rule, and because
the withheld band IS the finding (see tn36 below). Consuming it requires reading the
warning and opting in deliberately, which is the difference between an offered second arm
and a laundered mask.

THE COST, STATED PLAINLY. tn36 row 1 is withheld. That is the band the whole exercise was
motivated by -- the progress bar the six perfect-`change_fidelity`-1.0 candidates model
instead of the playfield. On the winning-route corpus it is clean by a 100x margin (12 of
2636 changed cells inside the mask, 7 of 7 changing transitions surviving); on the random
corpus it shows the SAME 1.0000 / 60 -> 0 signature as lf52. The two verdicts are about
different corpora and both are real. Under the stated asymmetry the tie goes to withholding,
so this capture does NOT deliver a default mask that fixes the tn36 degeneracy. Reporting
that is the point; weakening the gate until tn36 passed would be fitting the audit to the
hypothesis, which is the failure mode this whole file exists to avoid.

Spec: REQ-ARC-WMTE-6010 / REQ-ARC-WMTE-6015 / REQ-ARC-WMTE-6017.
"""

from __future__ import annotations

import json
import os
from typing import Any

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
OUT_DIR = os.path.join(REPO, "results", "arc_hud_mask_capture_20260801")
MASK_DIR = os.path.join(OUT_DIR, "masks")

HOW_TO_APPLY = (
    "mask = np.zeros(tuple(logical_shape), bool); mask[hud_rows, :] = True; "
    "mask[:, hud_cols] = True; then arc_executable_world_model.apply_hud_mask(grid, mask) "
    "on BOTH sides of any exact-match comparison. hud_rows/hud_cols already exclude every "
    "withheld band, so materialising them is the safe default. An EMPTY mask is a REFUSAL, "
    "not an absence -- apply nothing and record mask_status as the reason. Do NOT fold "
    "conditionally_clean_bands in without reading their warning."
)

COORD_NOTE = (
    "Indices are LOGICAL-grid rows/cols -- the same space as Transition.grid and "
    "Transition.next_grid. No frame-coordinate mask and no logical_hud_mask() downsample is "
    "required; that is the whole reason this path is reconstructible where the A/B's was not."
)


def _corpus_evidence(band_audit: dict) -> dict:
    """The winning-route numbers a reviewer needs to weigh a withheld band, and no more."""
    rec = band_audit["A3_swallow_guard"]["record"]
    return {
        "reason": rec.get("reason"),
        "changed_cell_overlap": rec.get("changed_cell_overlap"),
        "raw_changing_transitions": rec.get("raw_changing_transitions"),
        "masked_changing_transitions": rec.get("masked_changing_transitions"),
        "n_changed_cells_inside_mask": rec.get("n_changed_cells_inside_mask"),
        "n_changed_cells_total": rec.get("n_changed_cells_total"),
    }


def main() -> int:
    with open(os.path.join(OUT_DIR, "per_game.json")) as fh:
        games = {g["game"]: g for g in json.load(fh)["games"]}
    with open(os.path.join(OUT_DIR, "cross_corpus_check.json")) as fh:
        xc_doc = json.load(fh)
    xc = {g["game"]: g for g in xc_doc["games"]}

    decisions: list[dict] = []
    for name, g in games.items():
        if g.get("status") != "ok":
            continue
        x = xc.get(name, {})
        rand = x.get("random_action_corpus", {}) or {}
        rand_clean = rand.get("clean")
        rand_check = rand.get("swallow_check") or {}

        stage1_accepted = [b for b in g["bands"] if b["decision"]["accepted"]]
        stage1_excluded = [b for b in g["bands"] if not b["decision"]["accepted"]]

        # The random-corpus check was run on the UNION mask. Every game here has at most one
        # accepted band, so union == band; recorded rather than assumed, so a future game
        # with two accepted bands cannot silently inherit a union verdict as a per-band one.
        union_is_single_band = len(stage1_accepted) <= 1

        gate_applies = bool(stage1_accepted)
        gate_passed = bool(gate_applies and rand_clean is True)

        if not gate_applies:
            final_accepted: list[dict] = []
            withheld: list[dict] = []
            status = g["mask_status"]
        elif gate_passed:
            final_accepted = stage1_accepted
            withheld = []
            status = "accepted_clean_on_both_corpora"
        else:
            final_accepted = []
            withheld = stage1_accepted
            status = "withheld_second_corpus_swallow_refusal"

        rows = sorted(b["index"] for b in final_accepted if b["axis"] == "row")
        cols = sorted(b["index"] for b in final_accepted if b["axis"] == "col")
        shape = g["logical_shape"]
        masked_cells = len({("r", r) for r in rows}) * shape[1] + len(cols) * shape[0]
        if rows and cols:
            masked_cells -= len(rows) * len(cols)
        frac = round(masked_cells / float(shape[0] * shape[1]), 6)

        payload: dict[str, Any] = {
            "game": name,
            "stage": "final_after_second_corpus_gate",
            "supersedes": "the provisional mask file written by capture.py",
            "coordinate_space": "logical_grid",
            "coordinate_space_note": COORD_NOTE,
            "cell": g["cell"],
            "logical_shape": shape,
            "hud_rows": rows,
            "hud_cols": cols,
            "masked_cells": masked_cells,
            "masked_cell_fraction": frac,
            "mask_status": status,
            "second_corpus_gate": {
                "applies": gate_applies,
                "passed": gate_passed,
                "corpus": (
                    f"collect_transitions(n={xc_doc['corpus']['n']}, "
                    f"seed={xc_doc['corpus']['seed']}) -- random actions from reset"
                ),
                "clean": rand_clean,
                "reason": rand_check.get("reason"),
                "changed_cell_overlap": rand_check.get("changed_cell_overlap"),
                "raw_changing_transitions": rand_check.get("raw_changing_transitions"),
                "masked_changing_transitions": rand_check.get("masked_changing_transitions"),
                "union_mask_equals_single_band": union_is_single_band,
            },
            "same_band_as_explorer_edge_bar_detector": x.get("same_band_as_explorer"),
            "explorer_edge_bar_mask": x.get("explorer_edge_bar_mask"),
            "excluded_bands": [
                {
                    "axis": b["axis"],
                    "index": b["index"],
                    "changed_fraction": b["changed_fraction"],
                    "stage": "stage_1_audit",
                    "reason": b["decision"]["reason"],
                    "alarms": b["decision"]["alarms"],
                }
                for b in stage1_excluded
            ],
            "conditionally_clean_bands": [
                {
                    "axis": b["axis"],
                    "index": b["index"],
                    "changed_fraction": b["changed_fraction"],
                    "stage": "withheld_at_stage_2",
                    "stage_1_reason_accepted": b["decision"]["reason"],
                    "winning_route_corpus": _corpus_evidence(b["audit_full_trajectory"]),
                    "random_action_corpus": {
                        "reason": rand_check.get("reason"),
                        "changed_cell_overlap": rand_check.get("changed_cell_overlap"),
                        "raw_changing_transitions": rand_check.get("raw_changing_transitions"),
                        "masked_changing_transitions": rand_check.get(
                            "masked_changing_transitions"
                        ),
                    },
                    "refusal_class": (
                        "measured_swallow"
                        if rand_check.get("reason") == "mask_overlaps_majority_of_changed_cells"
                        else "unmeasurable_on_this_corpus"
                    ),
                    "warning": (
                        "NOT part of hud_rows/hud_cols. Clean on the winning-route corpus, "
                        "refused on the random-action corpus. Using it is an explicit "
                        "opt-in to a mask whose safety is corpus-conditional, and any result "
                        "obtained with it must report the refusal alongside."
                    ),
                }
                for b in withheld
            ],
            "how_to_apply": HOW_TO_APPLY,
        }
        with open(os.path.join(MASK_DIR, f"{name}.json"), "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        decisions.append(
            {
                "game": name,
                "mask_status": status,
                "hud_rows": rows,
                "hud_cols": cols,
                "masked_cell_fraction": frac,
                "second_corpus_gate_applies": gate_applies,
                "second_corpus_gate_passed": gate_passed,
                "second_corpus_reason": rand_check.get("reason"),
                "n_withheld_bands": len(withheld),
                "n_stage1_excluded_bands": len(stage1_excluded),
                "same_band_as_explorer": x.get("same_band_as_explorer"),
            }
        )

    index = {
        d["game"]: {
            "mask_file": f"masks/{d['game']}.json",
            "mask_status": d["mask_status"],
            "hud_rows": d["hud_rows"],
            "hud_cols": d["hud_cols"],
            "masked_cell_fraction": d["masked_cell_fraction"],
        }
        for d in decisions
    }
    with open(os.path.join(MASK_DIR, "_index.json"), "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(os.path.join(OUT_DIR, "final_decisions.json"), "w") as fh:
        json.dump({"decisions": decisions}, fh, indent=2, sort_keys=True)
        fh.write("\n")

    shipped = [d for d in decisions if d["hud_rows"] or d["hud_cols"]]
    withheld = [d for d in decisions if d["n_withheld_bands"]]
    print(
        json.dumps(
            {
                "n_games": len(decisions),
                "n_masks_shipped": len(shipped),
                "shipped": [d["game"] for d in shipped],
                "withheld_at_stage_2": [d["game"] for d in withheld],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
