#!/usr/bin/env python3
"""REQ-ARC-WMTE-6244: characterize Mode A's error class (Phase 4 of the 2026-08-08 ARC live-agent
improvement plan, redirected -- see the plan doc's Phase 4a correction for why 4a's original "retrain
the CNN" premise was superseded before this ran).

WHAT MODE A IS. `docs/research-notes/arc-world-model-admission-is-the-bottleneck-2026-07-29.md`
split 33 measured world-model rows into two structural failure modes. Mode A (12/33 rows;
ar25, ka59, re86, cd82-on-changed-frames): `correct_changed_cells=0, spurious_changed_cells=0` --
the induced engine predicts nothing ever changes, on frames where the real game DID change. That
note's own "next question worth asking", stated explicitly as a CPU-only follow-up needing no GPU
time, is answered here: "is the induced engine missing the action semantics, the object identity,
or the update rule?"

METHOD. For each Mode A game, collect fresh transitions from the REAL offline game
(`e3.collect_transitions`, CPU-only, no LLM/GPU), load the FROZEN origin-fixture induced engine
(`e3.load_origin_fixture_engine`, matching what the admission-bottleneck note itself measured --
not the mutable store, which REQ-ARC-WMTE-6016 warns gets rewritten in place), and for every
transition where the real game DID change (`grid != next_grid`), classify the induced engine's
prediction into one of:

  - IDENTITY: predicts the input grid unchanged, byte-for-byte. The engine extracted no dynamics
    at all for this transition -- direct code-level confirmation of the note's own framing.
  - WRONG_LOCATION: predicts SOME change, but not at any of the cells that actually changed. The
    engine is writing to the wrong object/place -- an object-identity error.
  - RIGHT_LOCATION_WRONG_VALUE: predicts change at (a subset of) the cells that actually changed,
    but with the wrong resulting color/value there. The engine found the right object but computed
    the wrong update -- an update-rule error.
  - PARTIAL_CORRECT: predicts change at some correct cells with correct values, but misses others
    entirely (recall < 1.0, precision high) -- a coverage gap in an otherwise-right update rule.

Also reads each engine's raw Python source (already committed, no execution needed) for a
qualitative note: does it branch on `action` at all, does it reference any per-object/component
extraction, or is it structurally a bare `return grid`.

Sample size: n=40 collected transitions per game (offline sim, seconds each) is enough to surface
several real changed-frame examples per game without a long CPU run; this is a CHARACTERIZATION
task (per the note's own framing), not a claim requiring Sample-Size-Rigor-discipline statistical
power -- the categorical question ("which of four buckets") does not need a large-N estimate the
way a percentage-point claim would.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402

OUT = REPO / "results" / "experiment_6244_mode_a_error_class_diagnosis.json"
MODE_A_GAMES = ("ar25", "ka59", "re86", "cd82")
N_TRANSITIONS = 40
SEED = 6244

# Qualitative read of each game's frozen origin-fixture source (short files, read directly --
# not re-derived from a re-run, since the code text itself is the evidence). See each game's
# entry in the artifact's "qualitative_finding" field for the one-line summary; the reasoning
# behind each is recorded here so a future reader can verify it against the same source.
QUALITATIVE_FINDINGS = {
    "ar25": (
        "Genuinely the most sophisticated of the four -- attempts real block-sliding physics "
        "(contiguous-color-run detection, per-row/column shift with boundary clamping) for "
        "actions 2/3/4. Crashes with a Python TypeError on most calls: "
        "`new_row[new_start:new_start+length] = color` assigns a SCALAR into a list SLICE "
        "(needs an iterable of matching length, e.g. `[color]*length`), and `new_row` is a "
        "plain Python list, not a numpy array. A mechanical code-generation bug, not a design "
        "or prediction-quality problem -- the underlying logic may well be closer to correct "
        "than the other three games' engines once this is fixed. NOT hand-patched here: this "
        "is an LLM-induced, mutable artifact, and hand-editing one frozen fixture would only "
        "fix this ONE copy, not the induction mechanism that generates the next one -- outer-"
        "loop RE on the induced code, not a fix to the live self-discovery process."
    ),
    "ka59": (
        "Branches on action (6/4/1) but every branch is gated behind a HARDCODED absolute-"
        "coordinate equality check (`if grid[30, 18] == 0:`, `if grid[27, 21] == 0:`) copied "
        "from the exact induce-time examples. Zero parametric/relative logic (no 'find the "
        "object and move it by direction' -- only 'if this exact pixel is exactly this exact "
        "value, write these exact other pixels'). Overfits to the induce-time transitions with "
        "no generalization capacity at all; that is why it is IDENTITY on 30/30 fresh "
        "transitions despite superficially 'branching on action'."
    ),
    "re86": (
        "Attempts genuine dynamics for actions 2/3 (a toggle-adjacent-cells idea for action 3, "
        "a directional-move idea for action 2) -- plausible READING of what the mechanic might "
        "be. But BOTH branches gate on `if data is not None:` before doing anything, assuming "
        "actions 2/3 always arrive as CLICK actions carrying (x, y) coordinate data. Verified "
        "directly: actions 2 and 3 WERE exercised in the collected sample (7 and 4 times "
        "respectively out of 40), every single time with `data=None` -- i.e. re86 exercises "
        "these as KEYBOARD-style actions in practice, not click actions, so the induced "
        "engine's core assumption about the action's MODALITY is wrong and the logic never "
        "fires. A genuine action-semantics error, distinct from ka59's overfitting and cd82's "
        "give-up pattern."
    ),
    "cd82": (
        "Keyboard actions 1/3/5 are bare `return grid` -- no attempt at modeling them at all "
        "(distinct from re86's wrong-assumption pattern: cd82 does not even try). Click action "
        "6 unconditionally writes color 5 at the clicked coordinate regardless of game state -- "
        "explains the 11/33 WRONG_LOCATION hits (a context-free, state-blind write). Checked "
        "whether the shipped structured-nav template (`InducedNavWorldModel`, REQ-ARC-WMTE-"
        "5842) already covers cd82 as a side effect (see this row's own "
        "`structured_nav_template_check` field for the exact numbers from THIS run's proper "
        "train/held-out split): `is_confident_nav` reports True, but held-out exact-match "
        "accuracy on the SAME fit is 0 -- reproduced across two independent transition samples "
        "at two different sizes/seeds (0/8 here; a prior ad hoc n=80 check found 0/14 with a "
        "completely different, empty displacement table on a different train slice). The "
        "instability itself (the SAME game's fit qualitatively disagreeing across samples) is "
        "the finding: `is_confident_nav` has a real false-positive gap on cd82, distinct from "
        "the sk48/wa30 cases its own docstring already documents excluding; cd82 is NOT a "
        "hidden nav-template win."
    ),
}


def _classify(pred: np.ndarray, grid: np.ndarray, next_grid: np.ndarray) -> str:
    if pred.shape != next_grid.shape:
        return "SHAPE_MISMATCH"
    if np.array_equal(pred, grid):
        return "IDENTITY"
    true_changed = grid != next_grid
    pred_changed = pred != grid
    overlap = true_changed & pred_changed
    if not overlap.any():
        return "WRONG_LOCATION"
    correct_at_overlap = pred[overlap] == next_grid[overlap]
    if correct_at_overlap.all() and pred_changed.sum() >= true_changed.sum():
        return "PARTIAL_CORRECT" if pred_changed.sum() < true_changed.sum() else "CORRECT"
    if correct_at_overlap.all():
        return "PARTIAL_CORRECT"
    return "RIGHT_LOCATION_WRONG_VALUE"


def _source_signals(src: str) -> dict:
    return {
        "n_lines": src.count("\n") + 1,
        "branches_on_action": bool(re.search(r"\baction\s*(==|in|!=)", src)),
        "references_action_param_at_all": "action" in src,
        "has_bare_return_grid_only": bool(
            re.fullmatch(
                r"[^\S\n]*def\s+engine\([^)]*\)[^\n]*:\s*\n[^\S\n]*return\s+grid\s*\.?\s*copy\(\)?\s*\n?",
                src.strip() + "\n",
            )
        )
        or src.strip().count("\n") < 3,
        "mentions_object_or_component": bool(
            re.search(r"\bobject|component|connected|blob|sprite\b", src, re.IGNORECASE)
        ),
    }


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_artifact() -> dict:
    t0 = time.time()
    per_game = []
    for game in MODE_A_GAMES:
        row: dict = {"game": game}
        try:
            transitions, _cell = e3.collect_transitions(game, n=N_TRANSITIONS, seed=SEED)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"collect_transitions failed: {exc!r}"
            per_game.append(row)
            continue
        try:
            engine, _is_lc = e3.load_origin_fixture_engine(game)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"load_origin_fixture_engine failed: {exc!r}"
            per_game.append(row)
            continue

        src_path = e3.E3_ORIGIN_FIXTURES_DIR / game / "world_model.py"
        src = src_path.read_text()
        row["source_sha256"] = _sha(src)
        row["source_signals"] = _source_signals(src)
        row["qualitative_finding"] = QUALITATIVE_FINDINGS[game]

        changed = [
            t
            for t in transitions
            if not np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid))
        ]
        row["n_transitions_collected"] = len(transitions)
        row["n_changed_transitions"] = len(changed)
        if not changed:
            row["classification_counts"] = {}
            row["dominant_error_class"] = "NO_CHANGED_TRANSITIONS_COLLECTED"
            per_game.append(row)
            continue

        counts: dict = {}
        examples: dict = {}
        for t in changed:
            g = np.asarray(t.grid)
            g1 = np.asarray(t.next_grid)
            try:
                pred = np.asarray(engine(g.copy(), t.action, t.data))
            except Exception as exc:  # noqa: BLE001
                cls = f"ENGINE_RAISED:{type(exc).__name__}"
            else:
                cls = _classify(pred, g, g1)
            counts[cls] = counts.get(cls, 0) + 1
            examples.setdefault(cls, {"action": t.action, "n_true_changed": int((g != g1).sum())})

        row["classification_counts"] = counts
        row["dominant_error_class"] = max(counts, key=counts.get)
        row["examples_by_class"] = examples

        # Action-sensitivity probe: does the SAME grid under two DIFFERENT actions produce two
        # DIFFERENT predictions? If not, the engine is action-insensitive regardless of what the
        # per-transition classification above says.
        if len(changed) >= 2:
            g0 = np.asarray(changed[0].grid)
            preds_by_action = {}
            for t in changed[:6]:
                try:
                    preds_by_action[t.action] = np.asarray(
                        engine(g0.copy(), t.action, t.data)
                    ).tobytes()
                except Exception:  # noqa: BLE001
                    pass
            row["distinct_predictions_across_distinct_actions_on_same_grid"] = len(
                set(preds_by_action.values())
            )
            row["actions_probed"] = list(preds_by_action.keys())

        # Cheap follow-up check: does the ALREADY-SHIPPED structured-nav template
        # (InducedNavWorldModel, REQ-ARC-WMTE-5842) already cover this game as a side effect?
        # Checked with a PROPER train/held-out split (not fit-and-check-on-the-same-data, which
        # over-reads confidence) -- this is the discipline the confidence gate itself is meant
        # to enforce, applied here as an adversarial check on the gate's OWN output.
        try:
            from carnot.agentic.arc_nav_world_model import InducedNavWorldModel

            k = max(2, int(len(transitions) * 0.25))
            nav_train, nav_held = transitions[:-k], transitions[-k:]
            nav = InducedNavWorldModel.fit(nav_train)
            is_nav = (
                bool(getattr(nav, "displacement", None))
                and getattr(nav, "goal_color", None) is not None
            )
            confident = is_nav and nav.is_confident_nav(grid=nav_train[-1].next_grid)
            nav_check = {
                "is_nav": is_nav,
                "is_confident_nav": bool(confident),
                "displacement": {str(k): v for k, v in (nav.displacement or {}).items()},
                "n_move_transitions": nav.fit_quality.get("n_move_transitions"),
            }
            if confident:
                eng, _isdone = nav.as_callables()
                nav_changed = [
                    t
                    for t in nav_held
                    if not np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid))
                ]
                correct = sum(
                    1
                    for t in nav_changed
                    if np.array_equal(
                        np.asarray(eng(np.asarray(t.grid), t.action, t.data)),
                        np.asarray(t.next_grid),
                    )
                )
                nav_check["heldout_changed_transitions"] = len(nav_changed)
                nav_check["heldout_exact_match_correct"] = correct
            row["structured_nav_template_check"] = nav_check
        except Exception as exc:  # noqa: BLE001
            row["structured_nav_template_check"] = {"error": repr(exc)[:160]}

        per_game.append(row)

    art = {
        "experiment": "experiment_6244_mode_a_error_class_diagnosis",
        "title": "Mode A error-class characterization: action semantics, object identity, or update rule?",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": (
            "For each Mode A game (predicts-nothing-changes, per the 2026-07-29 admission-"
            "bottleneck note), on transitions where the real game DID change, does the induced "
            "engine's prediction fail because it never extracted the action semantics (IDENTITY), "
            "picked the wrong object (WRONG_LOCATION), or computed the wrong update for the right "
            "object (RIGHT_LOCATION_WRONG_VALUE)?"
        ),
        "mode_a_games": list(MODE_A_GAMES),
        "n_transitions_requested_per_game": N_TRANSITIONS,
        "per_game": per_game,
        "solve_provenance": "development_proxy",
        "solve_provenance_principle": (
            "reads frozen origin-fixture engines and offline-arcade transitions for analysis; no "
            "solve is claimed, no live agent action is taken."
        ),
        "arc_solve_claim": False,
        "verifier_is_oracle": True,
        "verifier_is_oracle_principle": (
            "the offline game's own next_grid IS the ground truth this classification compares "
            "against -- an execution-grounded characterization, not an oracle-distinct verifier claim."
        ),
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "inference_substrate_principle": (
            "pure Python offline-arcade stepping + reading already-committed engine source; no "
            "GGUF load, no CUDA, no LLM call anywhere in this script."
        ),
        "random_seed": SEED,
    }

    counts_by_class_all_games: dict = {}
    for row in per_game:
        for cls, n in row.get("classification_counts", {}).items():
            counts_by_class_all_games[cls] = counts_by_class_all_games.get(cls, 0) + n
    art["aggregate_classification_counts"] = counts_by_class_all_games

    dominant = [r.get("dominant_error_class") for r in per_game if r.get("dominant_error_class")]
    art["honest_verdict"] = "complete_mode_a_error_class_diagnosis_" + "_".join(
        f"{g}_{d}" for g, d in zip(MODE_A_GAMES, dominant, strict=False)
    )
    art["honest_verdict_principle"] = (
        "terminal `complete_` prefix; names every game's dominant error class in one string so no "
        "reader can cite an aggregate without seeing the per-game breakdown it came from."
    )

    art["duration_s"] = round(time.time() - t0, 3)
    payload = {k: v for k, v in art.items() if k not in ("duration_s",)}
    art["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return art


def main() -> int:
    art = build_artifact()
    OUT.write_text(json.dumps(art, indent=2, default=str) + "\n", encoding="utf-8")
    for row in art["per_game"]:
        print(
            row["game"],
            row.get("dominant_error_class"),
            row.get("classification_counts"),
            "action_sensitive_distinct_preds="
            + str(row.get("distinct_predictions_across_distinct_actions_on_same_grid")),
        )
    print("verdict:", art["honest_verdict"])
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
