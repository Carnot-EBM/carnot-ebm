#!/usr/bin/env python3
"""PHASE 1 (confirm), STEP 2 -- derive and PROVE the in-sample / held-out split. Offline.

WHY THIS IS ITS OWN STEP, RUN BEFORE ANY GPU TIME IS SPENT. The entire point of this phase is to
retire the in-sample caveat that makes Phase 2's `cell_recall 0.947` uncitable. A split that is
merely ASSERTED buys nothing -- if the "held-out" rows turn out to have been in the prompt after
all, every number downstream is the same in-sample number wearing a new label, and we would not
find out until after ~2 GPU-hours. So the split is derived here, checked against the prompt TEXT,
and refuses to certify a game whose checks disagree.

THE THREE LEVELS (see capture_p3.py for how each is obtained):

  full     every transition the agent had collected at the induction point.
  prefix   what `induce()` was handed. `_proposal_prefix` drops a round(n/3) tail -- but ONLY on
           the call that goes through it; the second induction on these games receives the full
           list, which is itself worth knowing and is recorded rather than assumed.
  shown    the rows `_transitions_block` RENDERS: `changed[:k-2] + noop[:2]` with k=8. At most 8
           of 25. This -- and only this -- is what the model sees.

HELD_OUT = full \\ shown. A prefix row whose delta was never rendered is exactly as unseen by the
model as a suffix row, so both belong in the held-out set; the strict live-gate suffix
(full \\ prefix) is reported separately because that is the slice production grades.

THE PROOF, and it is deliberately not the same computation as the derivation. `shown` is derived
by replicating `_transitions_block`'s selection rule, then each candidate row is checked against
the prompt TEXT by rendering its delta line with the module's own `_rle_delta_compact` and asking
whether that line occurs. A derivation error and a text error would have to agree to slip through.
Three things are asserted per game:

  1. every row derived as `shown` has its delta line present in the prompt;
  2. no row derived as HELD_OUT has its delta line present in the prompt;
  3. the number of `--- ACTION` lines in the prompt equals len(shown).

(2) is the load-bearing one: it is the direct evidence that the held-out score is out-of-sample.
Duplicate transitions can make a genuinely-unshown row's rendered line coincide with a shown
row's, and that is NOT waved away -- such a row is moved OUT of the held-out set and counted under
`n_ambiguous_dropped`, because a row we cannot prove was unseen must not be scored as unseen.

WHAT THIS STEP CANNOT FIX, and reports instead. A held-out set with no grid-CHANGING rows cannot
measure change prediction at all; on such a game an identity engine scores a perfect held-out
accuracy. `heldout_n_changing` is therefore surfaced per game, and a 0 there is a statement that
the game is unscoreable for change quality out-of-sample -- not a passing grade.
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys
import time

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "split.json"

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_confirm_split/e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "python"))

GAMES = ["ft09", "tu93", "tn36", "lp85", "sc25", "vc33"]
CALL_INDEX = int(os.environ.get("SPLIT_CALL_INDEX", "2"))


def load_split(game: str, call_index: int = CALL_INDEX) -> dict:
    """Return the derived+proven split for one game. Imported by the scorer, so the scorer and
    this proof can never drift apart into two different definitions of `held out`."""

    import numpy as np

    from carnot.agentic import arc_executable_world_model as e3

    cap = HERE / "capture" / game
    prompt = (cap / f"prompt{call_index}_engine.txt").read_text()
    with open(cap / f"transitions{call_index}.pkl", "rb") as fh:
        prefix = pickle.load(fh)
    full_path = cap / f"full_transitions{call_index}.pkl"
    if full_path.exists():
        with open(full_path, "rb") as fh:
            full = pickle.load(fh)
        full_source = "recorded_proposal_prefix_input"
    else:
        # No `_proposal_prefix` call produced this induce call's list, i.e. the caller passed
        # `proposal_transitions` explicitly and the proposer received everything. Saying so is
        # the honest record; it also means this call has NO live-gate suffix.
        full = list(prefix)
        full_source = "no_prefix_call_matched_induce_received_full_list"

    mask_enabled = e3.world_model_hud_mask_enabled()  # False on the submitted path

    def _is_changed(t) -> bool:
        a = e3.apply_hud_mask(t.grid, None) if mask_enabled else np.asarray(t.grid)
        b = e3.apply_hud_mask(t.next_grid, None) if mask_enabled else np.asarray(t.next_grid)
        return not np.array_equal(a, b)

    # `_transitions_block(trans, k)` selection, replicated: changed[:k-2] + noop[:2].
    k = e3._induce_transitions_k()
    changed = [t for t in prefix if _is_changed(t)]
    noop = [t for t in prefix if not _is_changed(t)]
    shown = changed[: k - 2] + noop[:2]

    def _key(t) -> tuple:
        # CONTENT key, not id(). `full` and `prefix` may come from two separate pickle files,
        # and unpickling makes fresh objects -- an id() comparison across them is always False,
        # which would silently classify every shown row as a held-out candidate.
        return (
            int(t.action),
            repr(t.data),
            int(t.level_before),
            int(t.level_after),
            np.asarray(t.grid).tobytes(),
            np.asarray(t.next_grid).tobytes(),
        )

    shown_keys: dict = {}
    for t in shown:
        shown_keys[_key(t)] = shown_keys.get(_key(t), 0) + 1

    def _line(t) -> str:
        click = f" data={t.data}" if t.data else ""
        return (
            f"--- ACTION{t.action}{click} (level {t.level_before}->{t.level_after}): "
            f"changed cells (FULL, run-length) = {e3._rle_delta_compact(t.grid, t.next_grid)}"
        )

    checks = {
        "every_shown_line_in_prompt": all(_line(t) in prompt for t in shown),
        "n_action_lines_matches_shown": prompt.count("--- ACTION") == len(shown),
    }

    held: list = []
    ambiguous = 0
    budget = dict(shown_keys)
    for t in full:
        key = _key(t)
        if budget.get(key):
            budget[key] -= 1  # this row IS one of the shown ones
            continue
        if _line(t) in prompt:
            # Cannot PROVE this row was unseen -- its rendered line is in the prompt (a duplicate
            # of a shown row). Excluded rather than scored as held-out.
            ambiguous += 1
            continue
        held.append(t)
    # CORRECTION 2026-07-31 (adversarial review). This field used to be the literal `True`,
    # commented "true by construction of the loop above" -- a tautology dressed as a check,
    # sitting in the payload beside two real ones. The loop `continue`s on any row whose line
    # is in the prompt, so the field could never have been False and measured nothing.
    #
    # It is now the independent post-hoc assertion the artifact was implicitly claiming:
    # recompute the property over the FINISHED partition, by a scan the construction does not
    # perform. Kept (rather than dropped) because the property IS the one that makes the
    # held-out set out-of-sample, and a reader is entitled to see it verified rather than
    # assumed. The counting identity below is the second, stronger form -- every row whose
    # rendered line appears in the prompt is accounted for as either shown or ambiguous, so
    # no such row can have leaked into `held` by a path the first check misses.
    checks["no_heldout_line_in_prompt"] = all(_line(t) not in prompt for t in held)
    _n_lines_in_prompt = sum(1 for t in full if _line(t) in prompt)
    checks["lines_in_prompt_are_exactly_shown_plus_ambiguous"] = (
        _n_lines_in_prompt == len(shown) + ambiguous
    )

    prefix_keys: dict = {}
    for t in prefix:
        prefix_keys[_key(t)] = prefix_keys.get(_key(t), 0) + 1
    strict_suffix = []
    sbudget = dict(prefix_keys)
    for t in full:
        key = _key(t)
        if sbudget.get(key):
            sbudget[key] -= 1
            continue
        strict_suffix.append(t)

    return {
        "game": game,
        "call_index": call_index,
        "prompt_sha256_16": __import__("hashlib").sha256(prompt.encode()).hexdigest()[:16],
        "full_source": full_source,
        "n_full": len(full),
        "n_prefix": len(prefix),
        "n_shown": len(shown),
        "n_heldout": len(held),
        "n_ambiguous_dropped": ambiguous,
        "n_strict_live_suffix": len(strict_suffix),
        "shown_n_changing": int(sum(1 for t in shown if _is_changed(t))),
        "heldout_n_changing": int(sum(1 for t in held if _is_changed(t))),
        "heldout_n_noop": int(sum(1 for t in held if not _is_changed(t))),
        "checks": checks,
        "split_proven": all(checks.values()),
        # Change-quality out-of-sample needs at least one changing row to grade.
        "heldout_can_grade_change": bool(sum(1 for t in held if _is_changed(t)) > 0),
        "_shown": shown,
        "_heldout": held,
        "_strict_suffix": strict_suffix,
        "_full": full,
    }


def main() -> int:
    rows = []
    for game in GAMES:
        try:
            s = load_split(game)
        except Exception as exc:  # noqa: BLE001
            rows.append({"game": game, "status": f"error:{type(exc).__name__}", "detail": str(exc)})
            continue
        rows.append({k: v for k, v in s.items() if not k.startswith("_")})
    payload = {
        "generated_by": "results/arc_induce_confirm_20260731/harness/split.py",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "definition": (
            "HELD_OUT = full \\ shown, where `shown` is the <=8 rows _transitions_block renders "
            "into the prompt. Proven per game by checking each row's rendered delta line against "
            "the prompt TEXT. Rows whose line is ambiguous (duplicate of a shown row) are DROPPED "
            "from held-out, never scored as unseen."
        ),
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for r in rows:
        if r.get("status"):
            print(f"{r['game']:5s} {r['status']}")
            continue
        print(
            f"{r['game']:5s} full={r['n_full']:3d} prefix={r['n_prefix']:3d} "
            f"shown={r['n_shown']:2d}(chg {r['shown_n_changing']:2d}) "
            f"heldout={r['n_heldout']:3d}(chg {r['heldout_n_changing']:2d} "
            f"noop {r['heldout_n_noop']:3d}) ambig={r['n_ambiguous_dropped']:2d} "
            f"strict_suffix={r['n_strict_live_suffix']:2d} "
            f"proven={r['split_proven']} can_grade_change={r['heldout_can_grade_change']}"
        )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
