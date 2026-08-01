#!/usr/bin/env python3
"""ONE game per process, KILLABLE. Rebuild the induction window the object-perception A/B used,
split it the way that run split it, rebuild the prompt, and record two things about the prompt
the frozen run never wrote down:

  1. WHERE THE LEVEL-UP ROW LANDED. `build_progress_window` returns a window that ENDS at the
     level-up transition (`_select_levelup_window` slices `trans[j-(k-1) : j+1]` around the LAST
     level-up), and `_split_prefix_heldout` takes the LAST third as held-out. So the one
     observed positive example of winning is structurally in the held-out half and cannot be in
     the prompt. This worker MEASURES that rather than arguing it.

  2. WHETHER THE `WIN TRANSITION` BLOCK IS PRESENT. `_transitions_block` emits that block --
     the only place in the whole prompt system that shows the model a real, self-observed win
     event, added by the 2026-07-29 win-state correction -- iff some transition it is HANDED has
     `level_after > level_before`. It is handed `shown`.

REPRODUCTION GATE. The rebuilt prompt's sha256 must equal the `prompt_sha256` the frozen A/B
recorded for that game's `off` arm. Without that check this worker would be describing a prompt
of its own construction, not the one the model was actually sent, and every statement about
prompt content downstream would be an assumption. `run.py` refuses to build the artifact if any
game's sha disagrees.

NO LLM, NO GPU, NO GENERATION. Stepping the offline arcade is the only compute here; the env
vars below are set BEFORE any carnot import so a CPU-only pass can never contend for a GPU
another session on this shared machine owns.
"""

import hashlib
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Nothing here induces, but a stray write into the live engine store would corrupt evidence
# another session may be reading. Redirect it before the import that reads it.
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_goal_anatomy_e3_never_written")
sys.path.insert(0, "/home/ianblenke/github.com/ianblenke/carnot/python")
sys.path.insert(0, "/home/ianblenke/github.com/ianblenke/carnot/scripts")

with open(sys.argv[1]) as _jf:
    job = json.loads(_jf.read())
game = job["game"]
out = {"game": game}
try:
    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_world_model_trust_energy as wmte

    w = atp.build_progress_window(game)
    if w is None:
        out["status"] = "no_window"
    else:
        win, _full, cell = w
        win = list(win)
        shown, held = wmte._split_prefix_heldout(win)
        # The A/B's `off` arm is the object-perception flag UNSET. Match it exactly.
        os.environ.pop("CARNOT_ARC_OBJECT_PERCEPTION", None)
        p_off = e3.induce_prompt(game, shown, cell)
        out.update(
            {
                "status": "ok",
                "n_window": len(win),
                "n_shown": len(shown),
                "n_heldout": len(held),
                "levelups_in_window": sum(1 for t in win if t.level_after > t.level_before),
                "levelups_in_shown": sum(1 for t in shown if t.level_after > t.level_before),
                "levelups_in_heldout": sum(1 for t in held if t.level_after > t.level_before),
                "win_transition_block_in_prompt": "WIN TRANSITION" in p_off,
                "opening_board_block_in_prompt": (
                    "BOARD AT THE START OF THE CURRENT LEVEL" in p_off
                ),
                "is_level_complete_mentions_in_prompt": p_off.count("is_level_complete"),
                "prompt_chars": len(p_off),
                "prompt_sha256": hashlib.sha256(p_off.encode()).hexdigest(),
            }
        )
except Exception as exc:  # noqa: BLE001 - a worker must record its failure, never raise it
    out["status"] = "error"
    out["error"] = repr(exc)[:400]
with open(sys.argv[2], "w") as _of:
    _of.write(json.dumps(out, indent=1))
