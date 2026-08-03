"""PREP (CPU only, no LLM): build every game's induction window, split it, and collect a LARGE
FRESH held-out set -- then PROVE, per row, that nothing scored was ever shown to the model.

WHY A SEPARATE FRESH SET AT ALL. The production split (`_split_prefix_heldout`, 1/3 tail) leaves
3-4 held-out rows per game, of which ~3 are verifier-gradable and changing. The brief's target is
`change_accuracy >= 0.5 on more than a handful of rows`, and 3 rows is not more than a handful:
at n=3 the metric can only take the values {0, 1/3, 2/3, 1}, so a single lucky exact match clears
0.5's neighbourhood and a 95% interval on it spans most of the unit line. The fresh set is
`collect_transitions` exploration of the OFFLINE sim from reset under a fixed seed -- rows that
were never rendered into any prompt because the prompt only ever contains the window prefix.

THE PURITY ARGUMENT, WHICH IS THE PRIMARY RISK IN THIS MEASUREMENT. A scored row is clean iff the
model could not have seen it. Two independent witnesses are computed for EVERY scored row, and
both must hold or the row is dropped and counted:

  1. CONTENT: the row's (grid, action, data, next_grid) digest does not equal that of any row in
     the SHOWN prefix. This catches an exploration row that happens to replay a shown transition.
  2. RENDERED LINE: the exact line `_transitions_block` would emit for the row -- built with the
     same `_rle_delta_compact` the prompt uses -- is not a substring of the induce prompt string
     that was actually sent. This is the `no_heldout_line_in_prompt` witness the 2026-08-01
     corpora recorded, and it is the stronger of the two because it tests the artifact the model
     received rather than a model of it.

Witness 2 is checked against the REAL prompt returned by `induce_prompt`, not against a
reconstruction of the selection rule. This project has twice been burned by two reconstructions
of one wrong formula agreeing with each other.

NOTHING HERE CALLS AN LLM. No GPU is touched. `results/arc_e3` is never written: E3_DIR is
redirected to this harness's own scratch before the import that reads it.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]

# E3_DIR is read at IMPORT time. results/arc_e3 is EVIDENCE -- read, never written.
SCRATCH = HERE / "e3_store"
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ["CARNOT_ARC_E3_DIR"] = str(SCRATCH)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # prep must never take a card

if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

# The frozen 20-game roster of the 2026-08-01 corpora, reused UNCHANGED. Reusing it rather than
# re-deriving one means the roster cannot have been chosen after seeing which games favour an arm.
ROSTER = [
    "ls20",
    "s5i5",
    "tu93",
    "cn04",
    "m0r0",
    "sk48",
    "ar25",
    "tr87",
    "g50t",
    "re86",
    "bp35",
    "sb26",
    "lf52",
    "su15",
    "lp85",
    "cd82",
    "wa30",
    "sc25",
    "tn36",
    "ka59",
]

FRESH_N = int(os.environ.get("ABL_FRESH_N", "220"))
FRESH_SEED = int(os.environ.get("ABL_FRESH_SEED", "20260802"))


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def row_digest(t) -> str:
    """Content identity of a transition. Includes both grids, the action and the click data, so
    two rows collide only if they are the same observation."""
    import numpy as np

    h = hashlib.sha256()
    for g in (np.asarray(t.grid), np.asarray(t.next_grid)):
        h.update(str(g.shape).encode())
        h.update(np.ascontiguousarray(g, dtype=np.int64).tobytes())
    h.update(repr((int(t.action), t.data)).encode())
    return h.hexdigest()


def main() -> int:
    t0 = time.time()
    import numpy as np
    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_world_model_trust_energy as wmte

    assert Path(e3.E3_DIR) == SCRATCH, f"E3_DIR isolation failed: {e3.E3_DIR}"

    def prompt_line(t) -> str:
        """The exact line `_transitions_block` emits for `t`, rendered with the prompt's own
        `_rle_delta_compact`. Used for MEMBERSHIP testing against the real prompt string."""
        click = f" data={t.data}" if t.data else ""
        return (
            f"--- ACTION{t.action}{click} (level {t.level_before}->{t.level_after}): "
            f"changed cells (FULL, run-length) = {e3._rle_delta_compact(t.grid, t.next_grid)}"
        )

    def gradable_changing(t) -> bool:
        """A row the verifier will actually grade on the change channel: not a level-up row
        (those are excluded from BOTH numerator and denominator) and not a no-op."""
        return t.level_after <= t.level_before and not np.array_equal(
            np.asarray(t.grid), np.asarray(t.next_grid)
        )

    meta: dict = {}
    store: dict = {}
    for game in ROSTER:
        t_g = time.time()
        try:
            w = atp.build_progress_window(game)
        except Exception as exc:  # noqa: BLE001
            meta[game] = {"built": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
            print(f"  {game}: WINDOW FAILED {type(exc).__name__}", flush=True)
            continue
        if w is None:
            meta[game] = {"built": False, "error": "no_window"}
            print(f"  {game}: no window", flush=True)
            continue
        win, _full, cell = w
        win = list(win)
        shown, tail = wmte._split_prefix_heldout(win)

        # The REAL prompt for this game -- the artifact the model receives. Every purity witness
        # below tests against this string, not against a re-derivation of the selection rule.
        prompt = e3.induce_prompt(game, shown, int(cell))
        shown_digests = {row_digest(t) for t in shown}

        # ---- FRESH held-out set: offline exploration the prompt never contained -------------
        fresh_raw, fresh_cell = [], None
        try:
            fresh_raw, fresh_cell = e3.collect_transitions(game, n=FRESH_N, seed=FRESH_SEED)
        except Exception as exc:  # noqa: BLE001
            meta_err = f"{type(exc).__name__}: {exc}"[:200]
            print(f"  {game}: collect_transitions FAILED {meta_err}", flush=True)
            meta_err_field = meta_err
        else:
            meta_err_field = None

        fresh, dropped_content, dropped_line = [], 0, 0
        for t in fresh_raw:
            if row_digest(t) in shown_digests:
                dropped_content += 1
                continue
            if prompt_line(t) in prompt:
                dropped_line += 1
                continue
            fresh.append(t)

        # The same two witnesses applied to the PRODUCTION tail. `_split_prefix_heldout` is a
        # disjoint slice so this should be vacuous -- it is checked anyway, because "should be"
        # is how a leak survives.
        tail_bad_content = sum(1 for t in tail if row_digest(t) in shown_digests)
        tail_bad_line = sum(1 for t in tail if prompt_line(t) in prompt)

        store[game] = {
            "shown": shown,
            "tail": tail,
            "fresh": fresh,
            "cell": int(cell),
            "prompt_sha256": sha(prompt),
        }
        meta[game] = {
            "built": True,
            "build_s": round(time.time() - t_g, 1),
            "cell": int(cell),
            "n_window": len(win),
            "n_shown": len(shown),
            "shown_n_changing": int(sum(1 for t in shown if gradable_changing(t))),
            "n_tail": len(tail),
            "tail_gradable_changing": int(sum(1 for t in tail if gradable_changing(t))),
            "n_fresh_collected": len(fresh_raw),
            "n_fresh_kept": len(fresh),
            "fresh_gradable_changing": int(sum(1 for t in fresh if gradable_changing(t))),
            "fresh_dropped_content_collision_with_shown": dropped_content,
            "fresh_dropped_rendered_line_in_prompt": dropped_line,
            "fresh_cell": (int(fresh_cell) if fresh_cell is not None else None),
            "fresh_cell_matches_window_cell": (fresh_cell == cell),
            "tail_rows_colliding_with_shown_content": tail_bad_content,
            "tail_rows_whose_line_is_in_prompt": tail_bad_line,
            "prompt_chars": len(prompt),
            "prompt_sha256": sha(prompt),
            "collect_error": meta_err_field,
        }
        print(
            f"  {game}: shown={len(shown)} tail={len(tail)}"
            f"(grad {meta[game]['tail_gradable_changing']}) "
            f"fresh={len(fresh)}(grad {meta[game]['fresh_gradable_changing']}) "
            f"drop[content={dropped_content} line={dropped_line}] "
            f"cellmatch={meta[game]['fresh_cell_matches_window_cell']} "
            f"{meta[game]['build_s']}s",
            flush=True,
        )

    with open(OUT / "windows.pkl", "wb") as fh:
        pickle.dump(store, fh)
    meta["_prep"] = {
        "roster": ROSTER,
        "n_built": sum(1 for g in ROSTER if meta.get(g, {}).get("built")),
        "fresh_n_requested": FRESH_N,
        "fresh_seed": FRESH_SEED,
        "duration_s": round(time.time() - t0, 1),
        "leak_check_definition": {
            "content": "sha256 over (grid, next_grid, action, data); a scored row must not equal "
            "any SHOWN row",
            "rendered_line": "the exact _transitions_block line, built with the prompt's own "
            "_rle_delta_compact, must not be a substring of the real induce_prompt "
            "string for that game",
            "on_violation": "the row is DROPPED from the scored set and counted; never scored",
        },
    }
    (OUT / "prep_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nprep done in {meta['_prep']['duration_s']}s -> {OUT / 'windows.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
