#!/usr/bin/env python3
"""REQ-ARC-WMTE-6245: an internal held-out consistency check does NOT discriminate real nav fits
from REQ-ARC-WMTE-6244's cd82 false positive (negative result; no code shipped).

MOTIVATION. REQ-ARC-WMTE-6244 found `is_confident_nav` passes cd82 (a game whose induced engine
gives up on keyboard actions and hardcodes a bad click response) with a real-looking 4-direction
displacement fit, yet that fit scores 0 held-out exact-match -- reproduced across two independent
samples. The gate's existing structural checks (no padding-colour avatar, >=3 directions, a real
goal colour) do not catch this. The first candidate fix tried here: refit on a TRAIN-only slice of
the same transitions and score the held slice's move-transitions -- a genuine rigid-avatar fit
should reproduce unseen moves of the SAME game; a spurious coincidental co-translation should not.

TWO SCORING METRICS TESTED, BOTH FAIL TO DISCRIMINATE.

1. Full-grid exact match: tu93 (the ONE source-verified real nav game) scores 0.0 on this
   internal check too -- NOT because the fit is wrong (avatar/displacement/goal all come out
   correct), but because of the SAME HUD/step-counter contamination this project's own prior work
   already diagnosed and worked around elsewhere (`structured_nav_heldout`'s own comment: "a
   CORRECT avatar-only nav model scores heldout ~0 ... it models the avatar's move, not the
   co-moving key / STEP-COUNTER HUD / rails"). A gate built on this metric would reject the one
   real nav game it must not reject.

2. Changed-cell recall (the metric the project already prefers for exactly this HUD-contamination
   reason, per `trust_cell_recall`): does NOT rescue the discrimination either. Measured:
   tu93=0.417, cd82=0.507, sk48=0.432, wa30=0.823 -- the two KNOWN false positives (cd82, wa30)
   score AS HIGH OR HIGHER than the one true positive. A threshold set to keep tu93 would also
   keep cd82 and wa30; a threshold set to reject cd82/wa30 would also reject tu93.

CONCLUSION. At n=40 transitions per game, an internal train/held-out split is too small and too
noisy to reliably separate a real rigid-avatar fit from a spurious coincidental co-translation --
under EITHER scoring metric tried. This is a genuine negative result for this specific approach,
not a implementation bug: both metrics were implemented correctly (verified: the fit's own
structural parameters for tu93's train-only refit are directly confirmed correct -- avatar={4,9},
displacement magnitudes match the known-good full-data fit, goal=14, wall={5} -- so the
discrimination failure is a property of the METRIC choice + sample size, not a fitting bug).
`is_confident_nav`'s existing three structural checks are UNCHANGED by this result; cd82's
false-positive gap (REQ-ARC-WMTE-6244) remains open, and the next candidate fix (if pursued) needs
either substantially more transitions per game than this project's live episode budgets typically
provide, or a genuinely different discriminating signal (not agreement, not internal held-out
accuracy under either metric tried here) than anything tested in REQ-ARC-WMTE-6244/6245.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402
from carnot.agentic.arc_nav_world_model import InducedNavWorldModel, _norm  # noqa: E402

OUT = REPO / "results" / "experiment_6245_nav_confidence_internal_heldout_negative.json"
GAMES = {
    "tu93": "known_true_positive",
    "cd82": "known_false_positive_req_6244",
    "sk48": "known_false_positive_req_5844_two_snake",
    "wa30": "known_false_positive_req_5844_sokoban",
}
N_TRANSITIONS = 40
SEED = 6244
HOLDOUT_FRAC = 0.25


def _internal_split_and_score(game: str) -> dict:
    trans, _cell = e3.collect_transitions(game, n=N_TRANSITIONS, seed=SEED)
    rows = [_norm(t) for t in trans]
    k = max(2, int(len(rows) * HOLDOUT_FRAC))
    train_rows, held_rows = rows[:-k], rows[-k:]
    train_model = InducedNavWorldModel.fit(train_rows)
    held_moves = [
        r
        for r in held_rows
        if r[1] in train_model.displacement
        and not np.array_equal(np.asarray(r[0]), np.asarray(r[2]))
    ]

    exact_matches = []
    cell_recalls = []
    for g0, a, g1, *_ in held_moves:
        g0a, g1a = np.asarray(g0), np.asarray(g1)
        pred = np.asarray(train_model.engine(g0, a))
        exact_matches.append(bool(pred.shape == g1a.shape and np.array_equal(pred, g1a)))
        m = g0a != g1a
        if pred.shape != g1a.shape or not m.any():
            cell_recalls.append(0.0)
        else:
            cell_recalls.append(float((pred[m] == g1a[m]).mean()))

    return {
        "train_fit_structural_params": {
            "avatar_colors": sorted(train_model.avatar_colors),
            "displacement": {str(k): v for k, v in train_model.displacement.items()},
            "goal_color": train_model.goal_color,
            "wall_colors": sorted(train_model.wall_colors),
        },
        "n_held_moves": len(held_moves),
        "internal_heldout_exact_match_rate": (
            round(float(np.mean(exact_matches)), 4) if exact_matches else None
        ),
        "internal_heldout_mean_cell_recall": (
            round(float(np.mean(cell_recalls)), 4) if cell_recalls else None
        ),
    }


def build_artifact() -> dict:
    t0 = time.time()
    per_game = {}
    for game, label in GAMES.items():
        row = {"role": label}
        try:
            row.update(_internal_split_and_score(game))
        except Exception as exc:  # noqa: BLE001
            row["error"] = repr(exc)[:200]
        per_game[game] = row

    exact_rates = {
        g: r["internal_heldout_exact_match_rate"]
        for g, r in per_game.items()
        if r.get("internal_heldout_exact_match_rate") is not None
    }
    recall_rates = {
        g: r["internal_heldout_mean_cell_recall"]
        for g, r in per_game.items()
        if r.get("internal_heldout_mean_cell_recall") is not None
    }
    tp_exact = exact_rates.get("tu93")
    fp_exact = [exact_rates[g] for g in ("cd82", "sk48", "wa30") if g in exact_rates]
    tp_recall = recall_rates.get("tu93")
    fp_recall = [recall_rates[g] for g in ("cd82", "sk48", "wa30") if g in recall_rates]

    exact_discriminates = bool(tp_exact is not None and fp_exact and tp_exact > max(fp_exact) + 0.2)
    recall_discriminates = bool(
        tp_recall is not None and fp_recall and tp_recall > max(fp_recall) + 0.2
    )

    art = {
        "experiment": "experiment_6245_nav_confidence_internal_heldout_negative",
        "title": (
            "Negative result: internal train/held-out consistency check does not discriminate "
            "real nav fits from REQ-ARC-WMTE-6244's cd82 false positive"
        ),
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": (
            "Does refitting InducedNavWorldModel on a train-only slice and scoring held-out "
            "move-transitions (via exact-match, or via changed-cell recall) reliably separate "
            "tu93 (the one known real nav fit) from cd82/sk48/wa30 (known false positives)?"
        ),
        "per_game": per_game,
        "exact_match_discriminates_tp_from_fp": exact_discriminates,
        "cell_recall_discriminates_tp_from_fp": recall_discriminates,
        "honest_verdict": (
            "complete_neither_metric_discriminates_no_code_shipped"
            if not exact_discriminates and not recall_discriminates
            else "complete_at_least_one_metric_discriminates_see_per_game_data"
        ),
        "honest_verdict_principle": (
            "terminal `complete_` prefix; states the negative result plainly rather than as a "
            "process-completion claim with the actual finding buried in prose."
        ),
        "solve_provenance": "development_proxy",
        "arc_solve_claim": False,
        "verifier_is_oracle": True,
        "verifier_is_oracle_principle": (
            "each game's own real next_grid is the ground truth this check compares against."
        ),
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "random_seed": SEED,
    }
    art["duration_s"] = round(time.time() - t0, 3)
    payload = {k: v for k, v in art.items() if k != "duration_s"}
    art["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return art


def main() -> int:
    art = build_artifact()
    OUT.write_text(json.dumps(art, indent=2, default=str) + "\n", encoding="utf-8")
    for game, row in art["per_game"].items():
        print(
            game,
            row.get("role"),
            "exact=",
            row.get("internal_heldout_exact_match_rate"),
            "cell_recall=",
            row.get("internal_heldout_mean_cell_recall"),
        )
    print("exact_match_discriminates:", art["exact_match_discriminates_tp_from_fp"])
    print("cell_recall_discriminates:", art["cell_recall_discriminates_tp_from_fp"])
    print("verdict:", art["honest_verdict"])
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
